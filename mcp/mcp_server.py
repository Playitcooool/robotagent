import asyncio
import json
import os
import struct
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# ======================
# 公共常量和工具函数
# ======================
MAX_STEPS_PER_CALL = 10000  # 单次工具调用最多步进数，防止超时
_sim_lock = threading.Lock()  # 保护 simulation_instance 全局状态
_plane_body_id = None  # 跟踪 plane 的 body_id，不假设为 0


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


def _clamp_vector(vec: list[float], lo: float, hi: float) -> list[float]:
    return [max(lo, min(hi, float(v))) for v in vec]


def _safe_step(cid: int, steps: int) -> int:
    """安全步进，防止超时。返回实际步进数。"""
    steps = max(1, min(int(steps), MAX_STEPS_PER_CALL))
    for _ in range(steps):
        p.stepSimulation(physicsClientId=cid)
    return steps


def _is_body_valid(cid: int, body_id: int) -> bool:
    """检查 body_id 是否对应一个有效物体。"""
    try:
        num = p.getNumBodies(physicsClientId=cid)
        return 0 <= body_id < num
    except Exception:
        return False


def _tool_error(task: str, msg: str, err_type: str = "error") -> dict:
    return {"task": task, "status": "error", "message": msg, "error_type": err_type}


# ======================
# 初始化 MCP 服务器
# ======================
mcp_server = FastMCP("pybullet_simulator")

# ======================
# PyBullet 环境管理
# ======================
# 模拟环境实例
simulation_instance = None

# 实时帧共享目录（默认放在当前 mcp 目录内，便于 Docker 挂载共享）
DEFAULT_STREAM_DIR = Path(__file__).resolve().parent / ".sim_stream"
STREAM_DIR = Path(
    os.environ.get("PYBULLET_STREAM_DIR", str(DEFAULT_STREAM_DIR))
).resolve()
LATEST_META_FILE = STREAM_DIR / "latest.json"
LATEST_FRAME_FILE = STREAM_DIR / "latest.png"
REQUIRED_PYBULLET_ASSETS = [
    "plane.urdf",
    "cube_small.urdf",
    "kuka_iiwa/model.urdf",
]


def _ensure_stream_dir():
    STREAM_DIR.mkdir(parents=True, exist_ok=True)


def _pybullet_asset_status() -> dict[str, bool]:
    try:
        data_path = Path(pybullet_data.getDataPath())
        return {
            rel: (data_path / rel).exists()
            for rel in REQUIRED_PYBULLET_ASSETS
        }
    except Exception:
        return {rel: False for rel in REQUIRED_PYBULLET_ASSETS}


def _write_json_atomic(path: Path, payload: dict[str, Any]):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Encode HxWx3 uint8 RGB array into PNG bytes without external deps."""
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB uint8 array")

    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(scanlines, level=6)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def _capture_rgb_frame(width: int = 320, height: int = 240) -> np.ndarray:
    cid = simulation_instance
    if cid is None or not p.isConnected(cid):
        # Return black frame if simulation is not connected
        return np.zeros((height, width, 3), dtype=np.uint8)
    aspect = width / max(1, height)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.2, 0.0, 0.02],
        distance=1.2,
        yaw=45.0,
        pitch=-35.0,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=cid,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=aspect, nearVal=0.1, farVal=10.0, physicsClientId=cid
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid,
    )
    rgba_array = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    return rgba_array[:, :, :3]


def _publish_realtime_frame(
    *,
    run_id: str,
    task: str,
    step_idx: int,
    total_steps: int,
    done: bool = False,
    extra: dict[str, Any] | None = None,
):
    _ensure_stream_dir()
    rgb = _capture_rgb_frame()
    png_bytes = _encode_png_rgb(rgb)

    tmp_frame = LATEST_FRAME_FILE.with_suffix(".png.tmp")
    with open(tmp_frame, "wb") as f:
        f.write(png_bytes)
    os.replace(tmp_frame, LATEST_FRAME_FILE)

    payload = {
        "run_id": run_id,
        "task": task,
        "step": int(step_idx),
        "total_steps": int(total_steps),
        "done": bool(done),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)

    _write_json_atomic(LATEST_META_FILE, payload)


def _publish_snapshot(task: str, *, done: bool = False, extra: dict[str, Any] | None = None):
    """Publish one snapshot frame for tools that do not iterate steps."""
    _publish_realtime_frame(
        run_id=str(uuid.uuid4()),
        task=task,
        step_idx=1,
        total_steps=1,
        done=done,
        extra=extra,
    )


def setup_simulation(gui: bool = False):
    """初始化PyBullet环境，只在首次创建时运行"""
    global simulation_instance, _plane_body_id
    with _sim_lock:
        if simulation_instance is None or not p.isConnected(simulation_instance):
            if gui:
                simulation_instance = p.connect(p.GUI)
            else:
                simulation_instance = p.connect(p.DIRECT)

            try:
                data_path_str = pybullet_data.getDataPath()
                p.setAdditionalSearchPath(data_path_str, physicsClientId=simulation_instance)
            except Exception:
                # pybullet_data 路径获取失败时继续，搜索路径可能已配置
                pass
            p.setGravity(0, 0, -9.8, physicsClientId=simulation_instance)
            _plane_body_id = p.loadURDF("plane.urdf", physicsClientId=simulation_instance)
            _ensure_stream_dir()
        else:
            print("PyBullet environment is already running.")


def ensure_simulation():
    if simulation_instance is None:
        setup_simulation(gui=False)


@contextmanager
def with_simulation():
    """Context manager that ensures simulation is initialized and yields the correct client ID.
    All PyBullet API calls MUST use this client ID to target the correct simulation instance."""
    ensure_simulation()
    yield simulation_instance


def cleanup_simulation():
    """关闭PyBullet环境，释放所有资源。"""
    global simulation_instance, _plane_body_id
    with _sim_lock:
        if simulation_instance is not None:
            try:
                if p.isConnected(simulation_instance):
                    p.disconnect(simulation_instance)
            except Exception:
                # Best-effort cleanup; always clear local state.
                pass
            simulation_instance = None
            _plane_body_id = None
            # 清理帧文件
            try:
                if LATEST_META_FILE.exists():
                    LATEST_META_FILE.unlink()
                if LATEST_FRAME_FILE.exists():
                    LATEST_FRAME_FILE.unlink()
            except Exception:
                pass
        else:
            print("No simulation environment to close.")


def step(n: int = 240, cid: int | None = None):
    """执行指定步数的仿真"""
    if cid is None:
        cid = simulation_instance
    _safe_step(cid, n)


# ======================
# Tool 1: 初始化模拟环境
# ======================
class InitializeSimulationArgs(BaseModel):
    gui: bool = Field(
        default=False,
        description=(
            "Whether to open PyBullet GUI. False is recommended for server/agent usage "
            "because DIRECT mode is faster and headless."
        ),
    )


@mcp_server.tool()
def initialize_simulation(args: InitializeSimulationArgs):
    """
    Initialize or reuse the global PyBullet world.

    Use this tool before any motion/control tool when the simulator may be uninitialized.
    Safe to call repeatedly; if a simulation is already running, it reuses the existing world.

    Returns:
    - task/status/message
    - physicsClientId: 当前连接的客户端ID（用于调试）
    - also publishes a snapshot frame to the realtime stream directory.

    When NOT to use:
    - Do not call this before every single action in the same episode; initialize once and reuse.
    - Do not use this as a cleanup/reset tool (use cleanup_simulation_tool when done).
    """
    try:
        setup_simulation(gui=args.gui)
        _publish_snapshot("initialize_simulation", done=False, extra={"status": "running"})
        asset_status = _pybullet_asset_status()

        with _sim_lock:
            cid = simulation_instance

        return {
            "task": "initialize_simulation",
            "status": "success",
            "message": "Simulation environment initialized and running.",
            "physicsClientId": cid,
            "asset_status": asset_status,
        }
    except Exception as e:
        return _tool_error("initialize_simulation", f"Failed to initialize: {e}", "pybullet")


@mcp_server.tool()
def check_static_assets():
    """
    Check required PyBullet static assets bundled by pybullet_data.

    Returns:
    - data_path
    - asset_status map
    - missing_assets list
    """
    try:
        data_path = Path(pybullet_data.getDataPath())
    except Exception as e:
        return {
            "task": "check_static_assets",
            "status": "error",
            "data_path": None,
            "asset_status": {},
            "missing_assets": REQUIRED_PYBULLET_ASSETS,
            "message": f"Failed to get pybullet_data path: {e}",
        }
    status = _pybullet_asset_status()
    missing = [k for k, v in status.items() if not v]
    return {
        "task": "check_static_assets",
        "status": "success" if not missing else "warning",
        "data_path": str(data_path),
        "asset_status": status,
        "missing_assets": missing,
    }


# ======================
# Tool 2: Push Cube Step-by-Step
# ======================
class PushCubeStepArgs(BaseModel):
    start_position: list[float] = Field(
        default=[0.0, 0.0, 0.02],
        description=(
            "Cube start position [x, y, z] in meters. "
            "Recommended z around 0.02 to place cube on plane."
        ),
    )
    push_vector: list[float] = Field(
        default=[0.2, 0.0, 0.0],
        description=(
            "Translation vector [dx, dy, dz] applied over the full episode. "
            "Positive x moves right in default camera view."
        ),
    )
    steps: int = Field(
        default=120,
        description=(
            "Number of incremental steps. Must be > 0. "
            "Higher values create smoother motion and denser frame stream."
        ),
    )


@mcp_server.tool()
def push_cube_step(args: PushCubeStepArgs):
    """
    Run a deterministic cube pushing sequence with realtime frame streaming.

    Best for simple motion-control demos and verifying scene updates.
    Creates a cube, moves it incrementally, and publishes frame metadata on each step.

    Returns:
    - final_position
    - object_id: 用于后续 get_object_state / delete_object
    - stream_id / stream_meta_path for frame tracking
    - human-readable message

    When NOT to use:
    - Do not use for articulated robot manipulation; this is a cube baseline.
    - Do not use with non-positive steps.
    """
    try:
        _validate_vector(args.start_position, "start_position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.push_vector, "push_vector", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("push_cube_step", str(e), "validation")

    steps = max(1, min(int(args.steps), MAX_STEPS_PER_CALL))

    try:
        with with_simulation() as cid:
            cube = p.loadURDF("cube_small.urdf", args.start_position, physicsClientId=cid)
            run_id = str(uuid.uuid4())

            # 线性插值推动
            start_pos = args.start_position
            dx = args.push_vector[0] / args.steps
            dy = args.push_vector[1] / args.steps
            dz = args.push_vector[2] / args.steps

            _publish_realtime_frame(
                run_id=run_id,
                task="push_cube_step",
                step_idx=0,
                total_steps=steps,
                done=False,
            )

            # 每次调用执行一个小步进
            for i in range(steps):
                new_pos = [
                    start_pos[0] + dx * i,
                    start_pos[1] + dy * i,
                    start_pos[2] + dz * i,
                ]
                p.resetBasePositionAndOrientation(cube, new_pos, [0, 0, 0, 1], physicsClientId=cid)
                _safe_step(cid, 1)
                _publish_realtime_frame(
                    run_id=run_id,
                    task="push_cube_step",
                    step_idx=i + 1,
                    total_steps=steps,
                    done=(i + 1 == steps),
                )

            final_pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId=cid)

        return {
            "task": "push_cube_step",
            "status": "success",
            "stream_id": run_id,
            "stream_meta_path": str(LATEST_META_FILE),
            "final_position": final_pos,
            "object_id": cube,
            "message": f"Cube pushed along vector {args.push_vector}.",
        }
    except Exception as e:
        return _tool_error("push_cube_step", f"PyBullet error: {e}", "pybullet")


# ======================
# Tool 3: Grab and Place Step-by-Step
# ======================
class GrabAndPlaceStepArgs(BaseModel):
    start_position: list[float] = Field(
        default=[0.2, 0.0, 0.02],
        description="Object start position [x, y, z] in meters.",
    )
    target_position: list[float] = Field(
        default=[0.4, 0.4, 0.02],
        description="Object target position [x, y, z] in meters.",
    )
    steps: int = Field(
        default=120,
        description=(
            "Placement phase step count (after a fixed lift phase). Must be > 0."
        ),
    )


@mcp_server.tool()
def grab_and_place_step(args: GrabAndPlaceStepArgs):
    """
    Simulate a simplified pick-and-place episode with realtime frame streaming.

    Workflow:
    1) Lift object to a transport pose.
    2) Move/place object to target.

    Returns final position and stream metadata for visualization.

    When NOT to use:
    - Do not use for precise grasp/contact planning; this is a simplified teleport-style routine.
    - Do not use with non-positive steps.
    """
    try:
        _validate_vector(args.start_position, "start_position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.target_position, "target_position", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("grab_and_place_step", str(e), "validation")

    steps = max(1, min(int(args.steps), MAX_STEPS_PER_CALL))

    try:
        with with_simulation() as cid:
            cube = p.loadURDF("cube_small.urdf", args.start_position, physicsClientId=cid)
            run_id = str(uuid.uuid4())
            total_steps = 60 + steps

            # Lift object (teleport)
            p.resetBasePositionAndOrientation(cube, [0, 0.2, 0.2], [0, 0, 0, 1], physicsClientId=cid)
            for i in range(60):
                _safe_step(cid, 1)
                _publish_realtime_frame(
                    run_id=run_id,
                    task="grab_and_place_step",
                    step_idx=i + 1,
                    total_steps=total_steps,
                    done=False,
                )

            # Place object at the target position
            p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1], physicsClientId=cid)
            for i in range(steps):
                _safe_step(cid, 1)
                current_step = 60 + i + 1
                _publish_realtime_frame(
                    run_id=run_id,
                    task="grab_and_place_step",
                    step_idx=current_step,
                    total_steps=total_steps,
                    done=(current_step == total_steps),
                )

            final_pos, _ = p.getBasePositionAndOrientation(cube, physicsClientId=cid)

        return {
            "task": "grab_and_place_step",
            "status": "success",
            "stream_id": run_id,
            "stream_meta_path": str(LATEST_META_FILE),
            "final_position": final_pos,
            "object_id": cube,
            "message": f"Object placed at target location {args.target_position}.",
        }
    except Exception as e:
        return _tool_error("grab_and_place_step", f"PyBullet error: {e}", "pybullet")


# 路径规划工具
class PathPlanningArgs(BaseModel):
    start_position: list[float] = Field(
        default=[0.2, 0.0, 0.02],
        description="Robot base start position [x, y, z] in meters.",
    )
    target_position: list[float] = Field(
        default=[0.4, 0.4, 0.02],
        description="Robot base target position [x, y, z] in meters.",
    )
    steps: int = Field(
        default=240,
        description=(
            "Linear interpolation steps from start to target. Must be > 0."
        ),
    )


@mcp_server.tool()
def path_planning(args: PathPlanningArgs):
    """
    Execute a simple path-planning baseline (linear path) for KUKA base motion.

    Intended as a lightweight planning/control placeholder, not full IK or obstacle-aware planning.
    Publishes frames at every step for downstream UI streaming.

    Returns final robot position and stream metadata.

    When NOT to use:
    - Do not use when obstacle avoidance or IK-level realism is required.
    - Do not use with non-positive steps.
    """
    try:
        _validate_vector(args.start_position, "start_position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.target_position, "target_position", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("path_planning", str(e), "validation")

    steps = max(1, min(int(args.steps), MAX_STEPS_PER_CALL))

    try:
        with with_simulation() as cid:
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=args.start_position, physicsClientId=cid)
            run_id = str(uuid.uuid4())

            # 假设这里的路径规划为线性移动
            move_vector = [
                (args.target_position[0] - args.start_position[0]) / args.steps,
                (args.target_position[1] - args.start_position[1]) / args.steps,
                (args.target_position[2] - args.start_position[2]) / args.steps,
            ]

            _publish_realtime_frame(
                run_id=run_id,
                task="path_planning",
                step_idx=0,
                total_steps=steps,
                done=False,
            )

            for i in range(steps):
                new_pos = [
                    args.start_position[0] + move_vector[0] * i,
                    args.start_position[1] + move_vector[1] * i,
                    args.start_position[2] + move_vector[2] * i,
                ]
                # Move the robot arm
                p.resetBasePositionAndOrientation(robot_id, new_pos, [0, 0, 0, 1], physicsClientId=cid)
                _safe_step(cid, 1)
                _publish_realtime_frame(
                    run_id=run_id,
                    task="path_planning",
                    step_idx=i + 1,
                    total_steps=steps,
                    done=(i + 1 == steps),
                )

            final_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)

        return {
            "task": "path_planning",
            "status": "success",
            "stream_id": run_id,
            "stream_meta_path": str(LATEST_META_FILE),
            "final_position": final_pos,
            "object_id": robot_id,
            "message": "Robot arm moved to target position.",
        }
    except Exception as e:
        return _tool_error("path_planning", f"PyBullet error: {e}", "pybullet")


# 增加摩擦力和弹性
class FrictionAndElasticityArgs(BaseModel):
    friction: float = Field(
        default=0.5,
        description=(
            "Lateral friction coefficient for test cube dynamics. Typical range: 0.0~2.0."
        ),
    )
    restitution: float = Field(
        default=0.9,
        description=(
            "Restitution (bounciness) for test cube. Typical range: 0.0~1.0."
        ),
    )


@mcp_server.tool()
def adjust_physics(args: FrictionAndElasticityArgs):
    """
    Create a cube and set its friction/restitution to test contact behavior.

    Use before control experiments that depend on surface interaction.
    Publishes a snapshot frame after applying dynamics.

    When NOT to use:
    - Do not use as a motion-control action; it only changes dynamics parameters.
    - Do not expect global scene-wide physics update; this currently applies to the created test cube.
    """
    friction = float(args.friction)
    restitution = float(args.restitution)
    if not (0.0 <= friction <= 10.0):
        return _tool_error("adjust_physics", "friction must be in [0.0, 10.0]", "validation")
    if not (0.0 <= restitution <= 1.0):
        return _tool_error("adjust_physics", "restitution must be in [0.0, 1.0]", "validation")

    try:
        with with_simulation() as cid:
            cube = p.loadURDF("cube_small.urdf", [0, 0, 0.02], physicsClientId=cid)

            # 调整摩擦力和弹性
            p.changeDynamics(
                cube, -1, lateralFriction=friction, restitution=restitution, physicsClientId=cid
            )
            _safe_step(cid, 1)
            _publish_snapshot(
                "adjust_physics",
                done=True,
                extra={"friction": friction, "restitution": restitution},
            )

            return {
                "task": "adjust_physics",
                "status": "success",
                "object_id": cube,
                "message": "Friction and elasticity adjusted for the cube.",
            }
    except Exception as e:
        return _tool_error("adjust_physics", f"PyBullet error: {e}", "pybullet")


# 多物体抓取工具
class MultiObjectGrabArgs(BaseModel):
    object_positions: list[list[float]] = Field(
        default=[[0.2, 0.0, 0.02], [0.4, 0.4, 0.02]],
        description="List of object start positions, each as [x, y, z].",
    )
    target_position: list[float] = Field(
        default=[0.6, 0.6, 0.02],
        description="Shared target position [x, y, z] for all objects.",
    )


@mcp_server.tool()
def multi_object_grab_and_place(args: MultiObjectGrabArgs):
    """
    Move multiple cubes to one target location in a single batch operation.

    Useful for multi-object scene setup and quick state generation.
    Publishes one snapshot frame after movement.

    When NOT to use:
    - Do not use for sequential pick-place with collision-aware planning.
    - Do not use when object-specific trajectories are required.
    """
    try:
        _validate_vector(args.target_position, "target_position", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("multi_object_grab_and_place", str(e), "validation")

    for i, pos in enumerate(args.object_positions):
        try:
            _validate_vector(pos, f"object_positions[{i}]", size=3, allow_negative=True, max_val=100.0)
        except ValueError as e:
            return _tool_error("multi_object_grab_and_place", str(e), "validation")

    try:
        with with_simulation() as cid:
            cubes = []
            errors = []
            for pos in args.object_positions:
                try:
                    cube = p.loadURDF("cube_small.urdf", pos, physicsClientId=cid)
                    cubes.append(cube)
                except Exception as e:
                    errors.append(f"Failed to load object at {pos}: {e}")

            # 抓取并移动到目标位置（单个失败不影响其他）
            for cube in cubes:
                try:
                    p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1], physicsClientId=cid)
                except Exception as e:
                    errors.append(f"Failed to reposition object {cube}: {e}")

            _safe_step(cid, 1)
            _publish_snapshot(
                "multi_object_grab_and_place",
                done=True,
                extra={"count": len(cubes), "errors": errors if errors else None},
            )

            return {
                "task": "multi_object_grab_and_place",
                "status": "success" if not errors else "warning",
                "object_ids": cubes,
                "target_position": args.target_position,
                "message": f"{len(cubes)} objects moved to the target position." + (f" ({len(errors)} errors)" if errors else ""),
            }
    except Exception as e:
        return _tool_error("multi_object_grab_and_place", f"PyBullet error: {e}", "pybullet")


# 模拟视觉传感器
class VisionSensorArgs(BaseModel):
    width: int = Field(
        default=640,
        description="Camera image width in pixels.",
    )
    height: int = Field(
        default=480,
        description="Camera image height in pixels.",
    )
    view_matrix: list[float] = Field(
        default=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        description=(
            "Flattened 4x4 view matrix used by PyBullet getCameraImage."
        ),
    )
    projection_matrix: list[float] = Field(
        default=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        description="Flattened 4x4 projection matrix for camera rendering.",
    )


@mcp_server.tool()
def simulate_vision_sensor(args: VisionSensorArgs):
    """
    Capture one camera image from the current scene.

    Note:
    - Returns raw image array from PyBullet (not PNG URL by default).
    - Also publishes a snapshot frame to the shared realtime stream for UI preview.

    When NOT to use:
    - Do not use if you only need object state/poses (use check_simulation_state instead).
    - Do not assume returned image is compressed/serialized for direct JSON transport.
    """
    width = max(1, min(int(args.width), 4096))
    height = max(1, min(int(args.height), 4096))
    try:
        _validate_vector(args.view_matrix, "view_matrix", size=16, allow_negative=True, max_val=1e6)
        _validate_vector(args.projection_matrix, "projection_matrix", size=16, allow_negative=True, max_val=1e6)
    except ValueError as e:
        return _tool_error("simulate_vision_sensor", str(e), "validation")

    try:
        with with_simulation() as cid:
            img_arr = p.getCameraImage(
                width,
                height,
                viewMatrix=args.view_matrix,
                projectionMatrix=args.projection_matrix,
                physicsClientId=cid,
            )
            _publish_snapshot("simulate_vision_sensor", done=True)
            return {
                "task": "simulate_vision_sensor",
                "status": "success",
                "image": img_arr[2],  # Return RGB image as base64 (or image path)
                "message": "Captured image from the vision sensor.",
            }
    except Exception as e:
        return _tool_error("simulate_vision_sensor", f"PyBullet error: {e}", "pybullet")


# ======================
# Tool 4: Cleanup Simulation
# ======================
@mcp_server.tool()
def cleanup_simulation_tool():
    """
    Tear down the active simulation connection and clear local simulation state.

    Use when a task is fully finished and resources should be released.

    When NOT to use:
    - Do not call mid-episode unless you intentionally want to end the current world.
    - Do not call before state-inspection tools that rely on an active simulation.
    """
    cleanup_simulation()
    return {
        "task": "cleanup_simulation",
        "status": "success",
        "message": "Simulation environment cleaned up.",
    }


@mcp_server.tool()
def check_simulation_state():
    """
    Inspect current simulation objects and return per-body base positions.

    Good for debugging scene state after control actions.
    Also publishes a snapshot frame so UI can refresh even for read-only checks.

    When NOT to use:
    - Do not use as a motion action; this is read-only inspection.
    - Do not expect detailed joint-level kinematics; it returns base positions only.
    """
    with with_simulation() as cid:
        num_objects = p.getNumBodies(physicsClientId=cid)
        state_data = {}

        # 获取所有物体的状态
        for obj_id in range(num_objects):
            pos, _ = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=cid
            )
            state_data[obj_id] = pos

        _publish_snapshot(
            "check_simulation_state",
            done=False,
            extra={"num_objects": num_objects},
        )

        return {
            "task": "check_simulation_state",
            "status": "success",
            "state_data": state_data,
            "message": "Simulation environment state checked.",
        }


# ======================
# Tool: Reset Simulation
# ======================
class ResetSimulationArgs(BaseModel):
    keep_objects: bool = Field(
        default=True,
        description="是否保留现有物体。True 保留物体但重置位置/速度，False 完全重置。",
    )


@mcp_server.tool()
def reset_simulation(args: ResetSimulationArgs):
    """
    重置仿真世界。

    可以选择完全重置或只重置物体位置和速度。

    Returns:
    - status: 操作结果
    - message: 描述信息
    """
    try:
        with with_simulation() as cid:
            removed_ids = []
            kept_ids = []
            if args.keep_objects:
                # 只重置位置和速度，保留物体
                num_bodies = p.getNumBodies(physicsClientId=cid)
                for body_id in range(num_bodies):
                    if body_id == _plane_body_id:
                        continue  # skip plane (use tracked ID)
                    try:
                        p.resetBasePositionAndOrientation(body_id, [0, 0, 0.5], [0, 0, 0, 1], physicsClientId=cid)
                        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid)
                        kept_ids.append(body_id)
                    except Exception as e:
                        kept_ids.append(f"error_{body_id}: {e}")
            else:
                # 完全重置 - 删除所有非地面物体
                num_bodies = p.getNumBodies(physicsClientId=cid)
                for body_id in range(num_bodies - 1, 0, -1):
                    if body_id == _plane_body_id:
                        continue
                    try:
                        p.removeBody(body_id, physicsClientId=cid)
                        removed_ids.append(body_id)
                    except Exception as e:
                        removed_ids.append(f"error_{body_id}: {e}")

        _publish_snapshot("reset_simulation", done=True, extra={"keep_objects": args.keep_objects})

        return {
            "task": "reset_simulation",
            "status": "success",
            "message": f"Simulation reset completed. Removed: {removed_ids}, Kept: {kept_ids}",
            "removed_ids": removed_ids,
            "kept_ids": kept_ids,
        }
    except Exception as e:
        return _tool_error("reset_simulation", f"PyBullet error: {e}", "pybullet")


# ======================
# Tool: Pause Simulation
# ======================
@mcp_server.tool()
def pause_simulation():
    """
    暂停仿真（NOTE: PyBullet 不支持真正暂停，此操作目前无效）。

    暂停后物理不再更新，常用于：
    - 执行检查操作
    - 批量设置状态
    - 调试和控制执行节奏
    """
    return {
        "task": "pause_simulation",
        "status": "warning",
        "message": "PyBullet does not support true pause in DIRECT mode. Use step_simulation with steps=0 to control timing manually.",
    }


# ======================
# Tool: Unpause Simulation
# ======================
@mcp_server.tool()
def unpause_simulation():
    """
    恢复仿真（NOTE: 此操作目前无效，因为 PyBullet 不支持暂停）。

    与 pause_simulation 配合使用，控制仿真节奏。
    """
    return {
        "task": "unpause_simulation",
        "status": "warning",
        "message": "PyBullet does not support true pause in DIRECT mode. No action taken.",
    }


# ======================
# Tool: Get Object State
# ======================
class GetObjectStateArgs(BaseModel):
    object_id: int = Field(
        description="物体 ID，从 check_simulation_state 获取",
    )


@mcp_server.tool()
def get_object_state(args: GetObjectStateArgs):
    """
    获取指定物体的详细状态。

    返回物体的位置、姿态、线速度和角速度。
    常用于：
    - 检查目标是否到达
    - 计算偏差
    - 验证任务完成

    Returns:
    - position: [x, y, z]
    - orientation: [x, y, z, w] (quaternion)
    - linear_velocity: [x, y, z]
    - angular_velocity: [x, y, z]
    """
    object_id = int(args.object_id)
    try:
        with with_simulation() as cid:
            if not _is_body_valid(cid, object_id):
                return {
                    "task": "get_object_state",
                    "status": "warning",
                    "object_id": object_id,
                    "message": f"Object ID {object_id} does not exist. Available IDs: 0 to {p.getNumBodies(physicsClientId=cid) - 1}",
                }
            pos, ori = p.getBasePositionAndOrientation(object_id, physicsClientId=cid)
            lin_vel, ang_vel = p.getBaseVelocity(object_id, physicsClientId=cid)
            return {
                "task": "get_object_state",
                "status": "success",
                "object_id": object_id,
                "position": list(pos),
                "orientation": list(ori),
                "linear_velocity": list(lin_vel),
                "angular_velocity": list(ang_vel),
                "message": f"Object {object_id} state retrieved.",
            }
    except Exception as e:
        return {
            "task": "get_object_state",
            "status": "error",
            "object_id": object_id,
            "message": f"Failed to get object state: {str(e)}",
        }


# ======================
# Tool: Set Object Position
# ======================
class SetObjectPositionArgs(BaseModel):
    object_id: int = Field(
        description="物体 ID",
    )
    position: list[float] = Field(
        default=[0.5, 0, 0.5],
        description="目标位置 [x, y, z]",
    )
    orientation: list[float] = Field(
        default=[0, 0, 0, 1],
        description="目标姿态四元数 [x, y, z, w]，默认朝上",
    )


@mcp_server.tool()
def set_object_position(args: SetObjectPositionArgs):
    """
    设置物体位置和姿态。

    原子化操作，用于：
    - 手动定位物体
    - 纠正位置偏差
    - 快速移动到目标点
    不需要物理仿真，直接设置。

    Returns:
    - position: 设置后的位置
    - orientation: 设置后的姿态
    """
    object_id = int(args.object_id)
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.orientation, "orientation", size=4, allow_negative=True, max_val=1.1)
    except ValueError as e:
        return _tool_error("set_object_position", str(e), "validation")

    try:
        with with_simulation() as cid:
            if not _is_body_valid(cid, object_id):
                return {
                    "task": "set_object_position",
                    "status": "warning",
                    "object_id": object_id,
                    "message": f"Object ID {object_id} does not exist. Available IDs: 0 to {p.getNumBodies(physicsClientId=cid) - 1}",
                }
            p.resetBasePositionAndOrientation(object_id, args.position, args.orientation, physicsClientId=cid)
            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

            _publish_snapshot("set_object_position", done=False, extra={"object_id": object_id})

            return {
                "task": "set_object_position",
                "status": "success",
                "object_id": object_id,
                "position": args.position,
                "orientation": args.orientation,
                "message": f"Object {object_id} moved to {args.position}",
            }
    except Exception as e:
        return {
            "task": "set_object_position",
            "status": "error",
            "object_id": object_id,
            "message": f"Failed to set object position: {str(e)}",
        }


# ======================
# Tool: Step Simulation (原子化步进)
# ======================
class StepSimulationArgs(BaseModel):
    steps: int = Field(
        default=1,
        description="执行的仿真步数",
    )


@mcp_server.tool()
def step_simulation(args: StepSimulationArgs):
    """
    执行指定步数的仿真。

    原子化操作，常用于：
    - 手动控制仿真进度
    - 每步检查状态
    - 实现闭环控制
    """
    steps = max(1, min(int(args.steps), MAX_STEPS_PER_CALL))
    try:
        with with_simulation() as cid:
            _safe_step(cid, steps)

        return {
            "task": "step_simulation",
            "status": "success",
            "steps": steps,
            "message": f"Executed {steps} simulation steps.",
        }
    except Exception as e:
        return _tool_error("step_simulation", f"PyBullet error: {e}", "pybullet")


# ======================
# Tool: Create Object
# ======================
class CreateObjectArgs(BaseModel):
    object_type: str = Field(
        default="cube",
        description="物体类型: cube, sphere, cylinder",
    )
    position: list[float] = Field(
        default=[0, 0, 0.5],
        description="物体位置 [x, y, z]",
    )
    size: list[float] = Field(
        default=[0.05, 0.05, 0.05],
        description="物体尺寸 [x, y, z] 或半径",
    )
    mass: float = Field(
        default=1.0,
        description="物体质量 (kg)",
    )
    color: list[float] = Field(
        default=[1, 0, 0, 1],
        description="RGBA 颜色",
    )


@mcp_server.tool()
def create_object(args: CreateObjectArgs):
    """
    创建单个物体。

    可创建 cube/sphere/cylinder，常用于：
    - 添加新物体
    - 创建不同形状的测试对象
    - 多物体场景搭建

    Returns:
    - object_id: 创建的物体 ID
    - position: 初始位置
    """
    # 校验 object_type
    geom_map = {"cube": p.GEOM_BOX, "sphere": p.GEOM_SPHERE, "cylinder": p.GEOM_CYLINDER}
    geom = geom_map.get(args.object_type.lower(), p.GEOM_BOX)
    actual_type = next((k for k, v in geom_map.items() if v == geom), "cube")

    # 校验 size
    try:
        _validate_vector(args.size, "size", size=3, allow_zero=False, allow_negative=False, max_val=10.0)
    except ValueError as e:
        return _tool_error("create_object", str(e), "validation")

    mass = float(args.mass)
    if mass <= 0:
        return _tool_error("create_object", "mass must be > 0", "validation")

    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.color, "color", size=4, allow_negative=False, max_val=1.0)
    except ValueError as e:
        return _tool_error("create_object", str(e), "validation")

    try:
        with with_simulation() as cid:
            half_extents = [s / 2 for s in args.size]
            if geom == p.GEOM_SPHERE:
                radius = args.size[0] / 2
                col_shape = p.createCollisionShape(geom, radius=radius, physicsClientId=cid)
                vis_shape = p.createVisualShape(geom, radius=radius, rgbaColor=args.color, physicsClientId=cid)
            elif geom == p.GEOM_CYLINDER:
                radius = args.size[0] / 2
                height = args.size[2]
                col_shape = p.createCollisionShape(geom, radius=radius, height=height, physicsClientId=cid)
                vis_shape = p.createVisualShape(geom, radius=radius, height=height, rgbaColor=args.color, physicsClientId=cid)
            else:
                col_shape = p.createCollisionShape(geom, halfExtents=half_extents, physicsClientId=cid)
                vis_shape = p.createVisualShape(geom, halfExtents=half_extents, rgbaColor=args.color, physicsClientId=cid)

            object_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=args.position,
                physicsClientId=cid,
            )

            _publish_snapshot("create_object", done=False, extra={"object_id": object_id})

            return {
                "task": "create_object",
                "status": "success",
                "object_id": object_id,
                "object_type": actual_type,
                "position": args.position,
                "message": f"Created {actual_type} at {args.position} with ID {object_id}",
            }
    except Exception as e:
        return _tool_error("create_object", f"Failed to create object: {e}", "creation")


# ======================
# Tool: Delete Object
# ======================
class DeleteObjectArgs(BaseModel):
    object_id: int = Field(
        description="要删除的物体 ID",
    )


@mcp_server.tool()
def delete_object(args: DeleteObjectArgs):
    """
    删除指定物体。

    常用于：
    - 清理不需要的物体
    - 重置场景

    Returns:
    - deleted_id: 被删除的物体 ID
    """
    object_id = int(args.object_id)
    try:
        with with_simulation() as cid:
            if not _is_body_valid(cid, object_id):
                return {
                    "task": "delete_object",
                    "status": "warning",
                    "object_id": object_id,
                    "message": f"Object ID {object_id} does not exist. Available IDs: 0 to {p.getNumBodies(physicsClientId=cid) - 1}",
                }
            p.removeBody(object_id, physicsClientId=cid)
            return {
                "task": "delete_object",
                "status": "success",
                "deleted_id": object_id,
                "message": f"Deleted object {object_id}",
            }
    except Exception as e:
        return {
            "task": "delete_object",
            "status": "error",
            "object_id": object_id,
            "message": f"Failed to delete object: {str(e)}",
        }


# ======================
# Tool: Get Simulation Info
# ======================
@mcp_server.tool()
def get_simulation_info():
    """
    获取仿真基本信息。

    返回：
    - timestep: 当前时间步
    - num_bodies: 物体数量
    - gravity: 重力设置
    - time_elapsed: 已用仿真时间
    """
    try:
        with with_simulation() as cid:
            params = p.getPhysicsEngineParameters(physicsClientId=cid)
            return {
                "task": "get_simulation_info",
                "status": "success",
                "timestep": params.get('fixedTimeStep', 0.0),
                "num_bodies": p.getNumBodies(physicsClientId=cid),
                "gravity": [
                    params.get('gravityAccelerationX', 0.0),
                    params.get('gravityAccelerationY', 0.0),
                    params.get('gravityAccelerationZ', -9.8),
                ],
                "message": "Simulation info retrieved.",
            }
    except Exception as e:
        return _tool_error("get_simulation_info", f"PyBullet error: {e}", "pybullet")


# ======================
# Tool: Set Gravity
# ======================
class SetGravityArgs(BaseModel):
    gravity: list[float] = Field(
        default=[0, 0, -9.8],
        description="重力向量 [x, y, z]",
    )


@mcp_server.tool()
def set_gravity(args: SetGravityArgs):
    """
    设置重力。

    常用于：
    - 模拟不同重力环境
    - 零重力实验
    - 月球/火星重力模拟
    """
    try:
        _validate_vector(args.gravity, "gravity", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("set_gravity", str(e), "validation")

    try:
        with with_simulation() as cid:
            p.setGravity(*args.gravity, physicsClientId=cid)

        return {
            "task": "set_gravity",
            "status": "success",
            "gravity": args.gravity,
            "message": f"Gravity set to {args.gravity}",
        }
    except Exception as e:
        return _tool_error("set_gravity", f"PyBullet error: {e}", "pybullet")


# ======================
# 启动 MCP 服务
# ======================
async def start_mcp_server():
    """显式定义异步启动函数，确保绑定 0.0.0.0"""
    # 调用异步启动方法，明确指定 host 和 port
    await mcp_server.run_http_async(
        host="0.0.0.0",  # 强制绑定所有网卡，外部可访问
        port=8000,  # 容器内端口（由 docker-compose 映射到宿主机 8001）
        transport="http",
    )


# ======================
# Run MCP Server
# ======================
if __name__ == "__main__":
    # 显式运行异步事件循环，避免隐式配置问题
    # 启动异步服务并阻塞，直到服务停止
    asyncio.run(start_mcp_server())
