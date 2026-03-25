import asyncio
import base64
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from fastmcp import FastMCP
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, GetEntityState, SetEntityState, SpawnEntity
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from pydantic import BaseModel, Field
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Empty


def _validate_vector(
    vec: list[float],
    name: str,
    size: int | None = None,
    *,
    allow_zero: bool = True,
    allow_negative: bool = True,
    max_val: float | None = None,
) -> list[float]:
    """校验向量参数，检测 NaN/Inf/超限值。"""
    if not isinstance(vec, (list, tuple)):
        raise ValueError(f"{name} must be a list, got {type(vec).__name__}")
    if size is not None and len(vec) != size:
        raise ValueError(f"{name} must have {size} elements, got {len(vec)}")
    for i, v in enumerate(vec):
        if not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{i}] must be numeric, got {type(v).__name__}")
        if np.isnan(v) or np.isinf(v):
            raise ValueError(f"{name}[{i}] must be finite, got {v}")
        if not allow_zero and v == 0:
            raise ValueError(f"{name}[{i}] must be non-zero, got {v}")
        if not allow_negative and v < 0:
            raise ValueError(f"{name}[{i}] must be non-negative, got {v}")
        if max_val is not None and abs(v) > max_val:
            raise ValueError(f"{name}[{i}] must be within [-{max_val}, {max_val}], got {v}")
    return vec


def _tool_error(task: str, msg: str) -> dict:
    return {"task": task, "status": "error", "message": msg}


# ======================
# 初始化 MCP 服务器
# ======================
mcp_server = FastMCP("gazebo_simulator")


DEFAULT_BUILTIN_MODEL_ROOT = Path(__file__).resolve().parent / "assets" / "gazebo_models"
BUILTIN_MODEL_ROOT = Path(
    os.environ.get("GAZEBO_BUILTIN_MODEL_ROOT", str(DEFAULT_BUILTIN_MODEL_ROOT))
).resolve()


def _available_builtin_models() -> list[str]:
    if not BUILTIN_MODEL_ROOT.exists():
        return []
    names: list[str] = []
    for p in sorted(BUILTIN_MODEL_ROOT.iterdir()):
        if p.is_dir() and (p / "model.sdf").exists():
            names.append(p.name)
    return names


def _resolve_builtin_model_path(name: str) -> Path | None:
    candidate = BUILTIN_MODEL_ROOT / name / "model.sdf"
    if candidate.exists():
        return candidate
    return None


# ======================
# ROS2 节点
# ======================
class GazeboMCPNode(Node):
    def __init__(self):
        super().__init__("gazebo_mcp_node")

        # Service clients
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.get_entity_client = self.create_client(GetEntityState, "/get_entity_state")
        self.set_entity_client = self.create_client(SetEntityState, "/set_entity_state")
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.reset_sim_client = self.create_client(Empty, "/reset_simulation")
        self.reset_world_client = self.create_client(Empty, "/reset_world")

        # State subscribers
        self._lock = threading.Lock()
        self._latest_model_states: ModelStates | None = None
        self._camera_frames: dict[str, bytes] = {}  # topic -> raw png bytes

        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, 10
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _model_states_cb(self, msg: ModelStates):
        with self._lock:
            self._latest_model_states = msg

    def _make_camera_cb(self, topic: str):
        def cb(msg: Image):
            # Convert sensor_msgs/Image to PNG bytes (RGB8 only for now)
            if msg.encoding not in ("rgb8", "bgr8"):
                return
            import struct, zlib  # local import to keep top-level clean

            h, w = msg.height, msg.width
            raw = bytes(msg.data)
            if msg.encoding == "bgr8":
                import numpy as np
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                arr = arr[:, :, ::-1]  # BGR -> RGB
                raw = arr.tobytes()

            # Encode as PNG
            def png_chunk(ctype, data):
                return (
                    struct.pack(">I", len(data))
                    + ctype
                    + data
                    + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
                )

            scanlines = b"".join(
                b"\x00" + raw[r * w * 3 : (r + 1) * w * 3] for r in range(h)
            )
            compressed = zlib.compress(scanlines, level=6)
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            png = (
                b"\x89PNG\r\n\x1a\n"
                + png_chunk(b"IHDR", ihdr)
                + png_chunk(b"IDAT", compressed)
                + png_chunk(b"IEND", b"")
            )
            with self._lock:
                self._camera_frames[topic] = png

        return cb

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_model_states(self) -> ModelStates | None:
        with self._lock:
            return self._latest_model_states

    def get_camera_frame(self, topic: str) -> bytes | None:
        with self._lock:
            return self._camera_frames.get(topic)

    def clear_state(self):
        """Clear all cached simulation state (model states and camera frames).
        Call this between episodes/queries to prevent cross-contamination."""
        with self._lock:
            self._latest_model_states = None
            self._camera_frames.clear()

    def wait_for_camera_frame(self, topic: str, timeout: float = 5.0) -> bytes | None:
        """等待相机帧，使用事件机制而非轮询"""
        import threading

        frame_ready = threading.Event()
        result = [None]  # 使用列表存储结果以在回调中修改

        def on_frame(msg: Image):
            if msg.encoding not in ("rgb8", "bgr8"):
                return
            import struct, zlib
            import numpy as np

            h, w = msg.height, msg.width
            raw = bytes(msg.data)
            if msg.encoding == "bgr8":
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                arr = arr[:, :, ::-1]
                raw = arr.tobytes()

            def png_chunk(ctype, data):
                return (
                    struct.pack(">I", len(data))
                    + ctype
                    + data
                    + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
                )

            scanlines = b"".join(
                b"\x00" + raw[r * w * 3 : (r + 1) * w * 3] for r in range(h)
            )
            compressed = zlib.compress(scanlines, level=6)
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            png = (
                b"\x89PNG\r\n\x1a\n"
                + png_chunk(b"IHDR", ihdr)
                + png_chunk(b"IDAT", compressed)
                + png_chunk(b"IEND", b"")
            )
            result[0] = png
            frame_ready.set()

        sub = self.create_subscription(Image, topic, on_frame, 1)
        try:
            if frame_ready.wait(timeout=timeout):
                return result[0]
            return None
        finally:
            # 清理subscription
            self.destroy_subscription(sub)

    # ------------------------------------------------------------------
    # Synchronous service call helper
    # ------------------------------------------------------------------
    def call_service_sync(self, client, request, timeout: float = 5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(
                f"Service '{client.srv_name}' not available within {timeout}s"
            )
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=timeout):
            raise TimeoutError(
                f"Service call to '{client.srv_name}' timed out after {timeout}s"
            )
        result = future.result()
        # 注意：某些ROS2服务可能正常返回空结果，不一定是错误
        return result


# ======================
# ROS2 全局单例
# ======================
_ros_node: GazeboMCPNode | None = None
_ros_executor: MultiThreadedExecutor | None = None
_ros_thread: threading.Thread | None = None


def ensure_ros() -> GazeboMCPNode:
    global _ros_node, _ros_executor, _ros_thread
    # Always create a fresh ROS node to avoid cross-query contamination.
    # Cleanup any existing node first.
    if _ros_node is not None or _ros_executor is not None or _ros_thread is not None:
        try:
            cleanup_ros()
        except Exception:
            pass

    if not rclpy.ok():
        rclpy.init()

    # Ensure Gazebo can resolve model:// references from built-in local assets.
    if BUILTIN_MODEL_ROOT.exists():
        existing = os.environ.get("GAZEBO_MODEL_PATH", "")
        model_path_parts = [p for p in existing.split(":") if p]
        if str(BUILTIN_MODEL_ROOT) not in model_path_parts:
            model_path_parts.insert(0, str(BUILTIN_MODEL_ROOT))
            os.environ["GAZEBO_MODEL_PATH"] = ":".join(model_path_parts)

    _ros_node = GazeboMCPNode()
    _ros_executor = MultiThreadedExecutor()
    _ros_executor.add_node(_ros_node)
    _ros_thread = threading.Thread(target=_ros_executor.spin, daemon=True)
    _ros_thread.start()
    return _ros_node


def cleanup_ros():
    global _ros_node, _ros_executor, _ros_thread
    if _ros_executor is not None:
        _ros_executor.shutdown(timeout_sec=2.0)
    if _ros_node is not None:
        _ros_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    _ros_node = None
    _ros_executor = None
    _ros_thread = None


# ======================
# Tool 1: 初始化 ROS2 连接
# ======================
@mcp_server.tool()
def initialize_ros_connection():
    """
    Initialize a fresh ROS2 node for Gazebo services/topics.

    Always creates a clean ROS2 node, shutting down any existing connection first.
    Safe to call at the start of each query to ensure a clean state.

    Returns:
    - task/status
    - node_name
    - message
    """
    node = ensure_ros()
    return {
        "task": "initialize_ros_connection",
        "status": "success",
        "node_name": node.get_name(),
        "message": "ROS2 node connected to Gazebo.",
    }


# ======================
# Tool 2: 生成模型
# ======================
class SpawnModelArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="Unique entity name in Gazebo world; must not conflict with existing model names.",
    )
    model_xml: str = Field(
        default="",
        description=(
            "URDF/SDF XML content. Use when model text is provided inline. "
            "If empty, model_path must be provided."
        ),
    )
    model_path: str = Field(
        default="",
        description=(
            "Absolute filesystem path to URDF/SDF file to load. "
            "Used only when model_xml is empty."
        ),
    )
    position: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="[x, y, z] spawn position in meters (world frame).",
    )
    orientation: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0],
        description="[x, y, z, w] quaternion orientation (world frame).",
    )
    robot_namespace: str = Field(
        default="",
        description="ROS2 namespace for spawned model plugins/topics (optional).",
    )


@mcp_server.tool()
def spawn_model(args: SpawnModelArgs):
    """
    Spawn a URDF/SDF model entity into Gazebo.

    Requires either model_xml or model_path. If both are empty, it will try
    built-in assets at `<mcp>/assets/gazebo_models/<model_name>/model.sdf`.

    Returns:
    - task/status
    - model_name
    - service status message

    When NOT to use:
    - Do not use for moving existing models (use set_model_state).
    - Do not call with duplicate model_name unless replacement behavior is explicitly desired.
    """
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=1000.0)
        _validate_vector(args.orientation, "orientation", size=4, allow_negative=True, max_val=1.1)
    except ValueError as e:
        return _tool_error("spawn_model", str(e))

    node = ensure_ros()

    xml = args.model_xml
    if not xml:
        path: Path | None = None
        if args.model_path:
            path = Path(args.model_path)
        else:
            path = _resolve_builtin_model_path(args.model_name)
        if path is None:
            builtins = _available_builtin_models()
            builtins_hint = f" Available built-ins: {builtins}." if builtins else ""
            return _tool_error(
                "spawn_model",
                f"Provide either model_xml or model_path."
                f"{builtins_hint} You can also set GAZEBO_BUILTIN_MODEL_ROOT.",
            )
        if not path.exists():
            return _tool_error("spawn_model", f"Model file not found: {path}")
        xml = path.read_text(encoding="utf-8")

    try:
        req = SpawnEntity.Request()
        req.name = args.model_name
        req.xml = xml
        req.robot_namespace = args.robot_namespace
        req.initial_pose = Pose(
            position=Point(
                x=args.position[0], y=args.position[1], z=args.position[2]
            ),
            orientation=Quaternion(
                x=args.orientation[0],
                y=args.orientation[1],
                z=args.orientation[2],
                w=args.orientation[3],
            ),
        )

        result = node.call_service_sync(node.spawn_client, req)
        return {
            "task": "spawn_model",
            "status": "success" if result.success else "failure",
            "model_name": args.model_name,
            "message": result.status_message,
        }
    except Exception as e:
        return _tool_error("spawn_model", f"Service call failed: {e}")


@mcp_server.tool()
def list_builtin_models():
    """
    List built-in Gazebo static models shipped with this MCP server.

    Returns:
    - root: model root directory
    - models: available built-in model names
    """
    return {
        "task": "list_builtin_models",
        "status": "success",
        "root": str(BUILTIN_MODEL_ROOT),
        "models": _available_builtin_models(),
    }


# ======================
# Tool 3: 删除模型
# ======================
class DeleteModelArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="Exact Gazebo model/entity name to delete.",
    )


@mcp_server.tool()
def delete_model(args: DeleteModelArgs):
    """
    Delete one model/entity from Gazebo by name.

    Returns:
    - task/status
    - model_name
    - service status message

    When NOT to use:
    - Do not use for clearing entire world (use reset_world/reset_simulation).
    - Do not call on unknown names without first checking list_models.
    """
    node = ensure_ros()
    req = DeleteEntity.Request()
    req.name = args.model_name
    result = node.call_service_sync(node.delete_client, req)
    return {
        "task": "delete_model",
        "status": "success" if result.success else "failure",
        "model_name": args.model_name,
        "message": result.status_message,
    }


# ======================
# Tool 4: 获取模型状态
# ======================
class GetModelStateArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="Gazebo model/entity name to query.",
    )
    reference_frame: str = Field(
        default="world",
        description="Reference frame for returned pose/twist (e.g., 'world').",
    )


@mcp_server.tool()
def get_model_state(args: GetModelStateArgs):
    """
    Query pose and twist (linear/angular velocity) of one Gazebo model.

    Returns:
    - position/orientation
    - linear_velocity/angular_velocity
    - task/status/model_name

    When NOT to use:
    - Do not use for continuous streaming state; this is a single snapshot call.
    - Do not use for listing all models (use list_models).
    """
    node = ensure_ros()
    req = GetEntityState.Request()
    req.name = args.model_name
    req.reference_frame = args.reference_frame
    result = node.call_service_sync(node.get_entity_client, req)

    pos = result.state.pose.position
    ori = result.state.pose.orientation
    lin = result.state.twist.linear
    ang = result.state.twist.angular

    return {
        "task": "get_model_state",
        "status": "success" if result.success else "failure",
        "model_name": args.model_name,
        "position": [pos.x, pos.y, pos.z],
        "orientation": [ori.x, ori.y, ori.z, ori.w],
        "linear_velocity": [lin.x, lin.y, lin.z],
        "angular_velocity": [ang.x, ang.y, ang.z],
    }


# ======================
# Tool 5: 设置模型状态
# ======================
class SetModelStateArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="Gazebo model/entity name to move.",
    )
    position: list[float] = Field(
        ...,
        description="[x, y, z] target position in meters.",
    )
    orientation: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0],
        description="[x, y, z, w] target orientation quaternion.",
    )
    linear_velocity: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="[vx, vy, vz] linear velocity in m/s.",
    )
    angular_velocity: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="[wx, wy, wz] angular velocity in rad/s.",
    )
    reference_frame: str = Field(
        default="world",
        description="Reference frame for the state command.",
    )


@mcp_server.tool()
def set_model_state(args: SetModelStateArgs):
    """
    Set model pose/twist immediately (teleport-style state update).

    Useful for scripted repositioning, resets, and deterministic test setup.

    Returns:
    - task/status
    - model_name
    - service status message

    When NOT to use:
    - Do not use as a physically realistic motion controller (this is instantaneous state set).
    - Do not use for spawning new models (use spawn_model).
    """
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=1000.0)
        _validate_vector(args.orientation, "orientation", size=4, allow_negative=True, max_val=1.1)
        _validate_vector(args.linear_velocity, "linear_velocity", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.angular_velocity, "angular_velocity", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("set_model_state", str(e))

    node = ensure_ros()
    try:
        req = SetEntityState.Request()
        req.state = EntityState(
            name=args.model_name,
            reference_frame=args.reference_frame,
            pose=Pose(
                position=Point(
                    x=args.position[0], y=args.position[1], z=args.position[2]
                ),
                orientation=Quaternion(
                    x=args.orientation[0],
                    y=args.orientation[1],
                    z=args.orientation[2],
                    w=args.orientation[3],
                ),
            ),
            twist=Twist(
                linear=Point(
                    x=args.linear_velocity[0],
                    y=args.linear_velocity[1],
                    z=args.linear_velocity[2],
                ),
                angular=Point(
                    x=args.angular_velocity[0],
                    y=args.angular_velocity[1],
                    z=args.angular_velocity[2],
                ),
            ),
        )
        result = node.call_service_sync(node.set_entity_client, req)
        return {
            "task": "set_model_state",
            "status": "success" if result.success else "failure",
            "model_name": args.model_name,
            "message": result.status_message,
        }
    except Exception as e:
        return _tool_error("set_model_state", f"Service call failed: {e}")


# ======================
# Tool 6: 列出所有模型
# ======================
@mcp_server.tool()
def list_models():
    """
    List all models currently reported in /model_states.

    Returns:
    - model_count
    - array of model name + coarse position

    When NOT to use:
    - Do not use when you need full twist/orientation per model (use get_model_state).
    - If /model_states is not publishing yet, result may be empty/warning.
    """
    node = ensure_ros()
    states = node.get_model_states()
    if states is None:
        return {
            "task": "list_models",
            "status": "warning",
            "models": [],
            "message": "No /model_states message received yet. Is gazebo_ros running?",
        }

    models = [
        {
            "name": name,
            "position": [
                states.pose[i].position.x,
                states.pose[i].position.y,
                states.pose[i].position.z,
            ],
        }
        for i, name in enumerate(states.name)
    ]
    return {
        "task": "list_models",
        "status": "success",
        "model_count": len(models),
        "models": models,
    }


# ======================
# Tool 7: 暂停仿真
# ======================
@mcp_server.tool()
def pause_simulation():
    """
    Pause Gazebo physics stepping.

    Use before deterministic state edits or for freezing the scene.

    When NOT to use:
    - Do not call if simulation is already paused unless idempotent behavior is acceptable.
    """
    node = ensure_ros()
    node.call_service_sync(node.pause_client, Empty.Request())
    return {"task": "pause_simulation", "status": "success", "message": "Simulation paused."}


# ======================
# Tool 8: 恢复仿真
# ======================
@mcp_server.tool()
def unpause_simulation():
    """
    Resume Gazebo physics stepping after pause.

    When NOT to use:
    - Do not call if simulation is already running unless idempotent behavior is acceptable.
    """
    node = ensure_ros()
    node.call_service_sync(node.unpause_client, Empty.Request())
    return {"task": "unpause_simulation", "status": "success", "message": "Simulation resumed."}


# ======================
# Tool 9: 重置仿真（时间+状态）
# ======================
@mcp_server.tool()
def reset_simulation():
    """
    Full simulation reset: reset simulation clock and model states.

    Use for hard episode restart.

    When NOT to use:
    - Do not use when you need to preserve simulation time (use reset_world).
    """
    node = ensure_ros()
    node.call_service_sync(node.reset_sim_client, Empty.Request())
    return {"task": "reset_simulation", "status": "success", "message": "Simulation reset."}


# ======================
# Tool 10: 重置世界（仅状态）
# ======================
@mcp_server.tool()
def reset_world():
    """
    World reset: reset model states while keeping simulation clock.

    Use for soft episode reset where time continuity matters.

    When NOT to use:
    - Do not use when you need full time reset (use reset_simulation).
    """
    node = ensure_ros()
    node.call_service_sync(node.reset_world_client, Empty.Request())
    return {"task": "reset_world", "status": "success", "message": "World reset."}


# ======================
# Tool 11: 捕获相机图像
# ======================
class CaptureCameraArgs(BaseModel):
    topic: str = Field(
        default="/camera/image_raw",
        description=(
            "ROS2 image topic to capture (must publish sensor_msgs/Image, rgb8/bgr8 preferred)."
        ),
    )
    timeout: float = Field(
        default=5.0,
        description="Max seconds to wait for one frame before returning timeout.",
    )


@mcp_server.tool()
def capture_camera(args: CaptureCameraArgs):
    """
    Capture one camera frame from a ROS2 image topic and return PNG base64.

    Returns:
    - status success/timeout
    - topic
    - image_base64 + format=png (on success)

    When NOT to use:
    - Do not use for high-rate continuous video streaming (this is single-frame polling).
    - Do not use on non-image topics.
    """
    node = ensure_ros()

    # 使用事件机制替代busy-wait，减少CPU占用
    # wait_for_camera_frame内部会创建subscription
    frame = node.wait_for_camera_frame(args.topic, timeout=args.timeout)
    if frame is not None:
        return {
            "task": "capture_camera",
            "status": "success",
            "topic": args.topic,
            "image_base64": base64.b64encode(frame).decode("utf-8"),
            "format": "png",
        }

    return {
        "task": "capture_camera",
        "status": "timeout",
        "topic": args.topic,
        "message": f"No frame received on '{args.topic}' within {args.timeout}s.",
    }


# ======================
# Tool 12: 清理 ROS2 连接
# ======================
@mcp_server.tool()
def cleanup_ros_connection():
    """
    Shutdown ROS2 node/executor and release all Gazebo MCP ROS resources.

    Use when agent session is finished and no further Gazebo tools are needed.

    When NOT to use:
    - Do not call mid-task before pending Gazebo calls complete.
    - Do not call if subsequent Gazebo tools are expected immediately (re-init required).
    """
    cleanup_ros()
    return {
        "task": "cleanup_ros_connection",
        "status": "success",
        "message": "ROS2 connection cleaned up.",
    }


# ======================
# Tool 12b: 清理仿真状态（保留连接）
# ======================
@mcp_server.tool()
def clear_simulation_state():
    """
    Clear cached simulation state (model states and camera frames) without shutting down the ROS connection.

    Use this between queries to prevent state from one query contaminating the next.
    The ROS2 connection and service clients remain alive.

    When NOT to use:
    - Do not use as a substitute for reset_simulation/reset_world (those reset Gazebo world state).
    - Use cleanup_ros_connection when the session is fully done.
    """
    node = ensure_ros()
    node.clear_state()
    return {
        "task": "clear_simulation_state",
        "status": "success",
        "message": "Cached simulation state cleared.",
    }


# ======================
# Tool 13: 获取仿真信息
# ======================
@mcp_server.tool()
def get_simulation_info():
    """
    获取 Gazebo 仿真基本信息。

    返回：
    - model_count: 当前模型数量
    - paused: 仿真是否暂停
    - topics: 常用 topic 列表

    When NOT to use:
    - Do not use for per-model detailed state (use get_model_state/list_models).
    """
    node = ensure_ros()
    states = node.get_model_states()

    return {
        "task": "get_simulation_info",
        "status": "success",
        "model_count": len(states.name) if states else 0,
        "paused": False,  # Gazebo 不直接提供，需要通过服务查询
        "message": "Simulation info retrieved.",
    }


# ======================
# Tool 14: 应用力到物体
# ======================
class ApplyForceArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="要施加力的模型名称",
    )
    force: list[float] = Field(
        default=[0, 0, 10],
        description="力向量 [Fx, Fy, Fz] 单位 N",
    )
    position: list[float] = Field(
        default=[0, 0, 0],
        description="施加点相对于模型中心的偏移 [x, y, z]",
    )


@mcp_server.tool()
def apply_force(args: ApplyForceArgs):
    """
    对模型施加外力（NOTE: 此工具尚未实现，始终返回 warning）。

    常用于：
    - 推动物体
    - 测试物理响应
    - 模拟碰撞

    When NOT to use:
    - Do not use for precise positioning (use set_model_state).
    - 此工具尚未实现，需要 gazebo_ros_pkgs 支持
    """
    try:
        _validate_vector(args.force, "force", size=3, allow_negative=True, max_val=10000.0)
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=1000.0)
    except ValueError as e:
        return _tool_error("apply_force", str(e))

    return {
        "task": "apply_force",
        "status": "warning",
        "model_name": args.model_name,
        "force": args.force,
        "message": "apply_force 未实现（需要 gazebo_ros_pkgs 的 /gazebo/ApplyBodyWrench 服务）。请使用 set_model_state 移动物体。",
    }


# ======================
# Tool 15: 移动物体 (原子化)
# ======================
class MoveObjectArgs(BaseModel):
    model_name: str = Field(
        ...,
        description="要移动的模型名称",
    )
    position: list[float] = Field(
        default=[0, 0, 0.5],
        description="目标位置 [x, y, z]",
    )
    orientation: list[float] = Field(
        default=[0, 0, 0, 1],
        description="目标姿态四元数 [x, y, z, w]",
    )


@mcp_server.tool()
def move_object(args: MoveObjectArgs):
    """
    原子化移动物体到目标位置。

    相当于 set_model_state 的简化版本，不设置速度。

    When NOT to use:
    - Do not use for continuous motion (use multiple set_model_state calls).
    """
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=1000.0)
        _validate_vector(args.orientation, "orientation", size=4, allow_negative=True, max_val=1.1)
    except ValueError as e:
        return _tool_error("move_object", str(e))

    # 构建 SetModelStateArgs 并调用 set_model_state
    new_args = SetModelStateArgs(
        model_name=args.model_name,
        position=args.position,
        orientation=args.orientation,
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        reference_frame="world",
    )
    return set_model_state(new_args)


# ======================
# Tool 16: 创建简单物体
# ======================
class CreateSimpleObjectArgs(BaseModel):
    name: str = Field(
        default="cube",
        description="物体名称",
    )
    shape: str = Field(
        default="box",
        description="形状: box, sphere, cylinder",
    )
    position: list[float] = Field(
        default=[0, 0, 0.5],
        description="位置 [x, y, z]",
    )
    size: list[float] = Field(
        default=[0.1, 0.1, 0.1],
        description="尺寸参数 (box: xyz, sphere: radius, cylinder: radius height)",
    )


@mcp_server.tool()
def create_simple_object(args: CreateSimpleObjectArgs):
    """
    创建简单的几何体。

    简化版的 spawn_model，直接使用内置几何形状。

    When NOT to use:
    - Do not use for complex URDF/SDF models (use spawn_model).
    """
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=1000.0)
        _validate_vector(args.size, "size", size=None, allow_zero=False, allow_negative=False, max_val=10.0)
    except ValueError as e:
        return _tool_error("create_simple_object", str(e))

    shape_type = args.shape.lower()
    if shape_type == "box":
        if len(args.size) < 3:
            return _tool_error("create_simple_object", "box requires 3 size values [x, y, z]")
        geom = f"<box><size>{args.size[0]} {args.size[1]} {args.size[2]}</size></box>"
    elif shape_type == "sphere":
        if len(args.size) < 1:
            return _tool_error("create_simple_object", "sphere requires at least 1 size value [radius]")
        geom = f"<sphere><radius>{args.size[0]}</radius></sphere>"
    elif shape_type == "cylinder":
        if len(args.size) < 2:
            return _tool_error("create_simple_object", "cylinder requires at least 2 size values [radius, height]")
        geom = f"<cylinder><radius>{args.size[0]}</radius><height>{args.size[1]}</height></cylinder>"
    else:
        return _tool_error("create_simple_object", f"Unknown shape: {shape_type}. Use box, sphere, or cylinder.")

    sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{args.name}'>
    <link name='link'>
      <collision name='collision'>
        <geometry>{geom}</geometry>
      </collision>
      <visual name='visual'>
        <geometry>{geom}</geometry>
      </visual>
    </link>
  </model>
</sdf>"""

    new_args = SpawnModelArgs(
        model_name=args.name,
        model_xml=sdf,
        model_path="",
        position=args.position,
        orientation=[0, 0, 0, 1],
        robot_namespace="",
    )
    return spawn_model(new_args)


# ======================
# 启动 MCP 服务
# ======================
async def start_mcp_server():
    await mcp_server.run_http_async(
        host="0.0.0.0",
        port=8002,
        transport="http",
    )


if __name__ == "__main__":
    asyncio.run(start_mcp_server())
