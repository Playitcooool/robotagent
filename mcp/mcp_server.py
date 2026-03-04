import asyncio
import json
import os
import struct
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from fastmcp import FastMCP
from pydantic import BaseModel, Field


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


def _ensure_stream_dir():
    STREAM_DIR.mkdir(parents=True, exist_ok=True)


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
    aspect = width / max(1, height)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.2, 0.0, 0.02],
        distance=1.2,
        yaw=45.0,
        pitch=-35.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=aspect, nearVal=0.1, farVal=10.0
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
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
    global simulation_instance
    if simulation_instance is None or not p.isConnected(simulation_instance):
        if gui:
            simulation_instance = p.connect(p.GUI)
        else:
            simulation_instance = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        _ensure_stream_dir()
    else:
        print("PyBullet environment is already running.")


def ensure_simulation():
    if simulation_instance is None:
        setup_simulation(gui=False)


def cleanup_simulation():
    """关闭PyBullet环境"""
    global simulation_instance
    if simulation_instance is not None:
        try:
            if p.isConnected(simulation_instance):
                p.resetSimulation(physicsClientId=simulation_instance)
                p.disconnect(simulation_instance)
        except Exception:
            # Best-effort cleanup; always clear local state.
            pass
        simulation_instance = None
    else:
        print("No simulation environment to close.")


def step(n: int = 240):
    """执行指定步数的仿真"""
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


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
    - also publishes a snapshot frame to the realtime stream directory.

    When NOT to use:
    - Do not call this before every single action in the same episode; initialize once and reuse.
    - Do not use this as a cleanup/reset tool (use cleanup_simulation_tool when done).
    """
    setup_simulation(gui=args.gui)
    _publish_snapshot("initialize_simulation", done=False, extra={"status": "running"})

    return {
        "task": "initialize_simulation",
        "status": "success",
        "message": "Simulation environment initialized and running.",
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
    - stream_id / stream_meta_path for frame tracking
    - human-readable message

    When NOT to use:
    - Do not use for articulated robot manipulation; this is a cube baseline.
    - Do not use with non-positive steps.
    """
    ensure_simulation()
    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    cube = p.loadURDF("cube_small.urdf", args.start_position)
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
        total_steps=args.steps,
        done=False,
    )

    # 每次调用执行一个小步进
    for i in range(args.steps):
        new_pos = [
            start_pos[0] + dx * i,
            start_pos[1] + dy * i,
            start_pos[2] + dz * i,
        ]
        p.resetBasePositionAndOrientation(cube, new_pos, [0, 0, 0, 1])
        step(1)
        _publish_realtime_frame(
            run_id=run_id,
            task="push_cube_step",
            step_idx=i + 1,
            total_steps=args.steps,
            done=(i + 1 == args.steps),
        )

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    return {
        "task": "push_cube_step",
        "status": "success",
        "stream_id": run_id,
        "stream_meta_path": str(LATEST_META_FILE),
        "final_position": final_pos,
        "message": f"Cube pushed along vector {args.push_vector}.",
    }


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
    ensure_simulation()
    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    cube = p.loadURDF("cube_small.urdf", args.start_position)
    run_id = str(uuid.uuid4())
    total_steps = 60 + args.steps

    # Lift object (teleport)
    p.resetBasePositionAndOrientation(cube, [0, 0.2, 0.2], [0, 0, 0, 1])
    for i in range(60):
        step(1)
        _publish_realtime_frame(
            run_id=run_id,
            task="grab_and_place_step",
            step_idx=i + 1,
            total_steps=total_steps,
            done=False,
        )

    # Place object at the target position
    p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1])
    for i in range(args.steps):
        step(1)
        current_step = 60 + i + 1
        _publish_realtime_frame(
            run_id=run_id,
            task="grab_and_place_step",
            step_idx=current_step,
            total_steps=total_steps,
            done=(current_step == total_steps),
        )

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    return {
        "task": "grab_and_place_step",
        "status": "success",
        "stream_id": run_id,
        "stream_meta_path": str(LATEST_META_FILE),
        "final_position": final_pos,
        "message": f"Object placed at target location {args.target_position}.",
    }


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
    ensure_simulation()
    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=args.start_position)
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
        total_steps=args.steps,
        done=False,
    )

    for i in range(args.steps):
        new_pos = [
            args.start_position[0] + move_vector[0] * i,
            args.start_position[1] + move_vector[1] * i,
            args.start_position[2] + move_vector[2] * i,
        ]
        # Move the robot arm
        p.resetBasePositionAndOrientation(robot_id, new_pos, [0, 0, 0, 1])
        step(1)
        _publish_realtime_frame(
            run_id=run_id,
            task="path_planning",
            step_idx=i + 1,
            total_steps=args.steps,
            done=(i + 1 == args.steps),
        )

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)

    return {
        "task": "path_planning",
        "status": "success",
        "stream_id": run_id,
        "stream_meta_path": str(LATEST_META_FILE),
        "final_position": final_pos,
        "message": "Robot arm moved to target position.",
    }


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
    ensure_simulation()
    cube = p.loadURDF("cube_small.urdf", [0, 0, 0.02])

    # 调整摩擦力和弹性
    p.changeDynamics(
        cube, -1, lateralFriction=args.friction, restitution=args.restitution
    )
    step(1)
    _publish_snapshot(
        "adjust_physics",
        done=True,
        extra={"friction": args.friction, "restitution": args.restitution},
    )

    return {
        "task": "adjust_physics",
        "status": "success",
        "message": "Friction and elasticity adjusted for the cube.",
    }


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
    ensure_simulation()
    cubes = []
    for pos in args.object_positions:
        cubes.append(p.loadURDF("cube_small.urdf", pos))

    # 抓取并移动到目标位置
    for cube in cubes:
        p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1])
    step(1)
    _publish_snapshot(
        "multi_object_grab_and_place",
        done=True,
        extra={"count": len(cubes)},
    )

    return {
        "task": "multi_object_grab_and_place",
        "status": "success",
        "message": "Multiple objects moved to the target position.",
    }


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
    ensure_simulation()
    width, height = args.width, args.height
    img_arr = p.getCameraImage(
        width,
        height,
        viewMatrix=args.view_matrix,
        projectionMatrix=args.projection_matrix,
    )
    _publish_snapshot("simulate_vision_sensor", done=True)
    return {
        "task": "simulate_vision_sensor",
        "status": "success",
        "image": img_arr[2],  # Return RGB image as base64 (or image path)
        "message": "Captured image from the vision sensor.",
    }


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
    ensure_simulation()
    num_objects = p.getNumBodies()  # 获取场景中的物体数量
    state_data = {}

    # 获取所有物体的状态
    for obj_id in range(num_objects):
        pos, _ = p.getBasePositionAndOrientation(
            obj_id
        )  # 获取物体的唯一 ID 并获取其位置和姿态
        state_data[obj_id] = pos  # 以物体 ID 为键，将位置存储在字典中

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
