import asyncio
import base64
import threading
import time
from typing import Any

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


# ======================
# 初始化 MCP 服务器
# ======================
mcp_server = FastMCP("gazebo_simulator")


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

    def subscribe_camera(self, topic: str):
        self.create_subscription(Image, topic, self._make_camera_cb(topic), 1)

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
        if result is None:
            raise RuntimeError(f"Service call to '{client.srv_name}' returned None")
        return result


# ======================
# ROS2 全局单例
# ======================
_ros_node: GazeboMCPNode | None = None
_ros_executor: MultiThreadedExecutor | None = None
_ros_thread: threading.Thread | None = None


def ensure_ros() -> GazeboMCPNode:
    global _ros_node, _ros_executor, _ros_thread
    if _ros_node is not None:
        return _ros_node

    if not rclpy.ok():
        rclpy.init()

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
    Initialize ROS2 node and connect to the running Gazebo instance.
    Must be called before using any other Gazebo tools.
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
    model_name: str = Field(..., description="Unique name for the model in the scene.")
    model_xml: str = Field(
        default="",
        description="URDF or SDF XML string. Leave empty to load from model_path.",
    )
    model_path: str = Field(
        default="", description="Absolute path to a URDF/SDF file."
    )
    position: list[float] = Field(
        default=[0.0, 0.0, 0.0], description="[x, y, z] spawn position in meters."
    )
    orientation: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0],
        description="[x, y, z, w] quaternion orientation.",
    )
    robot_namespace: str = Field(
        default="", description="ROS2 namespace for the spawned robot."
    )


@mcp_server.tool()
def spawn_model(args: SpawnModelArgs):
    """
    Spawn a URDF or SDF model into the running Gazebo simulation.
    """
    node = ensure_ros()

    xml = args.model_xml
    if not xml:
        if not args.model_path:
            raise ValueError("Provide either model_xml or model_path.")
        from pathlib import Path

        path = Path(args.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        xml = path.read_text(encoding="utf-8")

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


# ======================
# Tool 3: 删除模型
# ======================
class DeleteModelArgs(BaseModel):
    model_name: str = Field(..., description="Name of the model to remove.")


@mcp_server.tool()
def delete_model(args: DeleteModelArgs):
    """
    Delete a model from the Gazebo simulation by name.
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
    model_name: str = Field(..., description="Name of the model.")
    reference_frame: str = Field(
        default="world", description="Reference frame for the pose."
    )


@mcp_server.tool()
def get_model_state(args: GetModelStateArgs):
    """
    Get the current pose and twist (velocity) of a model in Gazebo.
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
    model_name: str = Field(..., description="Name of the model to move.")
    position: list[float] = Field(..., description="[x, y, z] target position.")
    orientation: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0],
        description="[x, y, z, w] target orientation quaternion.",
    )
    linear_velocity: list[float] = Field(
        default=[0.0, 0.0, 0.0], description="[vx, vy, vz] linear velocity."
    )
    angular_velocity: list[float] = Field(
        default=[0.0, 0.0, 0.0], description="[wx, wy, wz] angular velocity."
    )
    reference_frame: str = Field(
        default="world", description="Reference frame for the state."
    )


@mcp_server.tool()
def set_model_state(args: SetModelStateArgs):
    """
    Teleport a model to a given pose and optionally set its velocity.
    """
    node = ensure_ros()
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


# ======================
# Tool 6: 列出所有模型
# ======================
@mcp_server.tool()
def list_models():
    """
    List all models currently present in the Gazebo simulation.
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
    Pause the Gazebo physics engine.
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
    Resume the Gazebo physics engine after a pause.
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
    Reset the Gazebo simulation: resets both simulation time and all model states.
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
    Reset the Gazebo world: resets model states only, keeps simulation time.
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
        description="ROS2 image topic to capture from (must publish sensor_msgs/Image).",
    )
    timeout: float = Field(
        default=5.0, description="Max seconds to wait for a frame."
    )


@mcp_server.tool()
def capture_camera(args: CaptureCameraArgs):
    """
    Capture a single image frame from a Gazebo camera sensor via ROS2 topic.
    Returns the image as a base64-encoded PNG string.
    """
    node = ensure_ros()
    node.subscribe_camera(args.topic)

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        frame = node.get_camera_frame(args.topic)
        if frame is not None:
            return {
                "task": "capture_camera",
                "status": "success",
                "topic": args.topic,
                "image_base64": base64.b64encode(frame).decode("utf-8"),
                "format": "png",
            }
        time.sleep(0.05)

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
    Shut down the ROS2 node and clean up all resources.
    """
    cleanup_ros()
    return {
        "task": "cleanup_ros_connection",
        "status": "success",
        "message": "ROS2 connection cleaned up.",
    }


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
