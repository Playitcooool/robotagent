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
    tmp_id = f"{os.getpid()}.{threading.get_ident()}.{int(time.time() * 1000000)}"
    tmp_path = path.with_suffix(path.suffix + f".{tmp_id}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


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

    # Auto-frame: compute AABB of all objects for proper framing (includes robot arm height)
    num_bodies = p.getNumBodies(physicsClientId=cid)
    all_mins = []
    all_maxs = []
    for body_id in range(num_bodies):
        if body_id == _plane_body_id:
            continue  # Skip infinite ground plane
        try:
            aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=cid)
            all_mins.append(aabb_min)
            all_maxs.append(aabb_max)
        except Exception:
            continue
    if all_mins:
        scene_min = [min(m[i] for m in all_mins) for i in range(3)]
        scene_max = [max(m[i] for m in all_maxs) for i in range(3)]
        center = [(scene_min[i] + scene_max[i]) / 2 for i in range(3)]
        spread = max(scene_max[i] - scene_min[i] for i in range(3))
        distance = max(1.0, spread * 1.5)
    else:
        center = [0.0, 0.0, 0.2]
        distance = 1.5

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=center,
        distance=distance,
        yaw=45.0,
        pitch=-25.0,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=cid,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=aspect, nearVal=0.1, farVal=20.0, physicsClientId=cid
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
    try:
        rgb = _capture_rgb_frame()
    except Exception:
        return  # simulation not ready; skip silently
    png_bytes = _encode_png_rgb(rgb)

    ts = time.time()
    payload = {
        "run_id": run_id,
        "task": task,
        "step": int(step_idx),
        "total_steps": int(total_steps),
        "done": bool(done),
        "timestamp": ts,
    }
    if extra:
        payload.update(extra)

    # Use unique tmp file per call to avoid races between threads/tool calls
    tmp_id = f"{os.getpid()}.{threading.get_ident()}.{int(ts * 1000000)}"
    tmp_frame = LATEST_FRAME_FILE.with_suffix(f".png.{tmp_id}.tmp")
    try:
        with open(tmp_frame, "wb") as f:
            f.write(png_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_frame, LATEST_FRAME_FILE)
    except Exception:
        try:
            if tmp_frame.exists():
                tmp_frame.unlink()
        except Exception:
            pass
        return
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


# ======================
# Live frame streaming (background thread)
# ======================
_live_stream_thread: threading.Thread | None = None
_live_stream_stop: threading.Event = threading.Event()
_LIVE_STREAM_FPS = float(os.environ.get("PYBULLET_LIVE_FPS", "5"))


def _live_stream_loop():
    """Continuously publish frames at configured FPS while simulation is connected."""
    run_id = str(uuid.uuid4())
    interval = 1.0 / max(0.5, _LIVE_STREAM_FPS)
    print(f"[mcp_server] live stream thread started (fps={_LIVE_STREAM_FPS})")
    while not _live_stream_stop.is_set():
        try:
            cid = simulation_instance
            if cid is not None and p.isConnected(cid):
                _publish_realtime_frame(
                    run_id=run_id,
                    task="live",
                    step_idx=1,
                    total_steps=1,
                    done=False,
                    extra={"live": True},
                )
        except Exception as e:
            print(f"[mcp_server] live stream error: {e}")
        _live_stream_stop.wait(interval)
    print("[mcp_server] live stream thread stopped")


def _start_live_stream():
    global _live_stream_thread
    if _live_stream_thread is not None and _live_stream_thread.is_alive():
        return
    _live_stream_stop.clear()
    _live_stream_thread = threading.Thread(
        target=_live_stream_loop, daemon=True, name="pybullet-live-stream"
    )
    _live_stream_thread.start()


def _stop_live_stream():
    global _live_stream_thread
    _live_stream_stop.set()
    if _live_stream_thread is not None:
        _live_stream_thread.join(timeout=2.0)
        _live_stream_thread = None


def setup_simulation(gui: bool = False):
    """初始化PyBullet环境，每次调用都创建干净的环境。失败时抛出异常。"""
    global simulation_instance, _plane_body_id
    # Stop any existing live stream thread FIRST (before sim instance is torn down)
    _stop_live_stream()
    # Delete stale frame files so frontend doesn't see the previous task's scene
    try:
        if LATEST_META_FILE.exists():
            LATEST_META_FILE.unlink()
        if LATEST_FRAME_FILE.exists():
            LATEST_FRAME_FILE.unlink()
    except Exception:
        pass
    with _sim_lock:
        # 先清理旧环境（如果存在）
        if simulation_instance is not None:
            for attempt in range(3):
                try:
                    if p.isConnected(simulation_instance):
                        p.disconnect(simulation_instance)
                    break
                except Exception as e:
                    if attempt < 2:
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Failed to disconnect old PyBullet instance: {e}") from e
            simulation_instance = None
            _plane_body_id = None

        if gui:
            simulation_instance = p.connect(p.GUI)
        else:
            simulation_instance = p.connect(p.DIRECT)

        # 验证连接成功
        if not p.isConnected(simulation_instance):
            raise RuntimeError(f"PyBullet connection failed (client_id={simulation_instance})")

        try:
            data_path_str = pybullet_data.getDataPath()
            p.setAdditionalSearchPath(data_path_str, physicsClientId=simulation_instance)
        except Exception as e:
            raise RuntimeError(f"Failed to set PyBullet data path: {e}") from e
        p.setGravity(0, 0, -9.8, physicsClientId=simulation_instance)
        _plane_body_id = p.loadURDF("plane.urdf", physicsClientId=simulation_instance)
        if _plane_body_id < 0:
            raise RuntimeError("Failed to load plane.urdf")
        _ensure_stream_dir()
    # Start background live streaming thread (outside the lock to avoid deadlock)
    _start_live_stream()


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
    """关闭PyBullet环境，释放所有资源。重试disconnect直到成功。"""
    global simulation_instance, _plane_body_id
    # Stop live stream first (outside lock)
    _stop_live_stream()
    with _sim_lock:
        if simulation_instance is not None:
            # 重试disconnect，因为PyBullet偶发报错
            for attempt in range(3):
                try:
                    if p.isConnected(simulation_instance):
                        p.disconnect(simulation_instance)
                    break  # 成功断开
                except Exception as e:
                    if attempt < 2:
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    # 最后一次尝试失败，不再静默——让调用方知道清理失败
                    raise RuntimeError(f"Failed to disconnect PyBullet after 3 attempts: {e}") from e
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


def step(n: int = 240, cid: int | None = None):
    """执行指定步数的仿真"""
    if cid is None:
        cid = simulation_instance
    _safe_step(cid, n)


# ======================
# Tool 1: 初始化模拟环境
# ======================
class InitializeSimulationArgs(BaseModel):
    """Arguments for initialize_simulation. All fields optional."""
    gui: bool = Field(
        default=False,
        description=(
            "Whether to open PyBullet GUI. Default False (DIRECT mode, headless). "
            "Optional — can be omitted."
        ),
    )

    class Config:
        extra = "ignore"


@mcp_server.tool()
def initialize_simulation(args: InitializeSimulationArgs = InitializeSimulationArgs()):
    """
    Initialize a fresh PyBullet world.

    Always creates a clean simulation environment, disconnecting any existing
    PyBullet connection first. Safe to call at the start of each query to
    ensure a clean state.

    Returns:
    - task/status/message
    - physicsClientId: 当前连接的客户端ID（用于调试）
    - also publishes a snapshot frame to the realtime stream directory.

    Raises:
        Exception: If simulation initialization fails. The ToolRetryMiddleware
            will retry this call up to max_retries times.
    """
    if args.gui:
        print("[mcp_server] GUI mode requested but forcing DIRECT mode (no X server in container)")

    # Always do a fresh setup - ensures new tasks start with a clean scene.
    # Stale frame files are deleted inside setup_simulation→cleanup to prevent
    # frontend from showing previous task's screenshot.
    setup_simulation(gui=False)
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
# Tool 2-6: High-level demo tools REMOVED
# Previously: push_cube_step, grab_and_place_step, path_planning, adjust_physics,
# multi_object_grab_and_place. They self-created objects via loadURDF, incompatible
# with composition workflow. Use initialize_simulation → create_object → step_simulation
# → set_object_position primitives instead.
# ======================


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
        raise RuntimeError(f"simulate_vision_sensor failed: {e}") from e


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
        raise RuntimeError(f"reset_simulation failed: {e}") from e


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
        raise RuntimeError(f"get_object_state failed: {e}") from e


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
        raise RuntimeError(f"set_object_position failed: {e}") from e


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
            # Publish frames during stepping so the frontend sees continuous motion
            # (not a jump from initial to final state).
            run_id = str(uuid.uuid4())
            # Target ~30fps output. Physics at 240Hz default → 8 steps ~= 33ms sim time.
            steps_per_frame = max(1, min(8, steps // 20)) if steps >= 20 else 1
            done_count = 0
            while done_count < steps:
                batch = min(steps_per_frame, steps - done_count)
                _safe_step(cid, batch)
                done_count += batch
                _publish_realtime_frame(
                    run_id=run_id,
                    task="step_simulation",
                    step_idx=done_count,
                    total_steps=steps,
                    done=(done_count >= steps),
                )

        return {
            "task": "step_simulation",
            "status": "success",
            "steps": steps,
            "message": f"Executed {steps} simulation steps.",
        }
    except Exception as e:
        raise RuntimeError(f"step_simulation failed: {e}") from e


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

    # 校验 size (宽容：允许 1-3 元素，自动 pad 到 3)
    size_list = list(args.size) if isinstance(args.size, (list, tuple)) else [args.size]
    if len(size_list) == 1:
        size_list = [size_list[0]] * 3
    elif len(size_list) == 2:
        size_list = [size_list[0], size_list[0], size_list[1]]
    elif len(size_list) > 3:
        size_list = size_list[:3]
    args.size = size_list
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
        raise RuntimeError(f"create_object failed: {e}") from e


# ======================
# Tool: Load URDF Model (robots, articulated objects)
# ======================
class LoadUrdfArgs(BaseModel):
    urdf_path: str = Field(
        description=(
            "URDF file path. Common built-in models (pybullet_data):\n"
            "- 'kuka_iiwa/model.urdf' (7-DOF robot arm, KUKA IIWA)\n"
            "- 'franka_panda/panda.urdf' (Franka Panda arm)\n"
            "- 'r2d2.urdf' (R2D2 robot)\n"
            "- 'humanoid/humanoid.urdf'\n"
            "- 'husky/husky.urdf' (mobile base)"
        ),
    )
    position: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Base position [x, y, z] in meters.",
    )
    orientation: list[float] = Field(
        default=[0.0, 0.0, 0.0, 1.0],
        description="Base orientation as quaternion [x, y, z, w].",
    )
    use_fixed_base: bool = Field(
        default=True,
        description="Whether to fix the base to the world (true for robot arms).",
    )


@mcp_server.tool()
def load_urdf(args: LoadUrdfArgs):
    """
    Load a URDF model (robot arm, articulated body, etc.) into the simulation.

    Use this to visualize robot manipulators like KUKA IIWA or Franka Panda.
    Returns the body_id so subsequent tools (get_object_state, set_object_position,
    delete_object) can refer to it.

    Returns:
    - object_id: The body id for this URDF
    - num_joints: Number of joints in the model
    """
    try:
        _validate_vector(args.position, "position", size=3, allow_negative=True, max_val=100.0)
        _validate_vector(args.orientation, "orientation", size=4, allow_negative=True, max_val=1.1)
    except ValueError as e:
        return _tool_error("load_urdf", str(e), "validation")

    try:
        with with_simulation() as cid:
            body_id = p.loadURDF(
                args.urdf_path,
                basePosition=args.position,
                baseOrientation=args.orientation,
                useFixedBase=bool(args.use_fixed_base),
                physicsClientId=cid,
            )
            if body_id < 0:
                return _tool_error(
                    "load_urdf",
                    f"Failed to load URDF: {args.urdf_path}",
                    "load_failed",
                )
            num_joints = p.getNumJoints(body_id, physicsClientId=cid)
            _publish_snapshot("load_urdf", done=False, extra={"object_id": body_id})
            return {
                "task": "load_urdf",
                "status": "success",
                "object_id": body_id,
                "urdf_path": args.urdf_path,
                "position": args.position,
                "num_joints": int(num_joints),
                "message": f"Loaded {args.urdf_path} as object_id {body_id} with {num_joints} joints.",
            }
    except Exception as e:
        raise RuntimeError(f"load_urdf failed: {e}") from e


# ======================
# Tool: Set Joint Positions (robot arm control)
# ======================
class SetJointPositionsArgs(BaseModel):
    object_id: int = Field(description="Body ID of the robot (from load_urdf).")
    joint_positions: list[float] = Field(
        description=(
            "Target joint angles in radians. Length must match the robot's number of "
            "controllable joints. For KUKA IIWA: 7 values. For Franka Panda: 7 arm joints "
            "(ignore finger joints). Values typically in [-3.14, 3.14]."
        ),
    )
    max_force: float = Field(
        default=500.0,
        description="Maximum motor force applied to each joint.",
    )


@mcp_server.tool()
def set_joint_positions(args: SetJointPositionsArgs):
    """
    Set target joint positions for a robot arm using position control.

    This makes the robot arm MOVE to the specified joint configuration.
    After calling this, use step_simulation to let the physics engine
    actually move the joints (the arm will animate toward the target).

    Typical workflow:
    1. load_urdf → get robot object_id
    2. set_joint_positions(object_id, joint_positions=[...]) → set target
    3. step_simulation(steps=100) → arm moves to target (visible in live feed)
    4. get_object_state(object_id) → verify final position

    Returns:
    - joints_set: number of joints controlled
    """
    try:
        with with_simulation() as cid:
            if not _is_body_valid(cid, args.object_id):
                return _tool_error(
                    "set_joint_positions",
                    f"Object ID {args.object_id} not found.",
                    "validation",
                )
            num_joints = p.getNumJoints(args.object_id, physicsClientId=cid)
            # Find controllable (non-fixed) joints
            controllable = []
            for i in range(num_joints):
                info = p.getJointInfo(args.object_id, i, physicsClientId=cid)
                joint_type = info[2]
                if joint_type != p.JOINT_FIXED:
                    controllable.append(i)

            positions = args.joint_positions
            n = min(len(positions), len(controllable))
            for i in range(n):
                p.setJointMotorControl2(
                    bodyUniqueId=args.object_id,
                    jointIndex=controllable[i],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=positions[i],
                    force=args.max_force,
                    physicsClientId=cid,
                )

            _publish_snapshot("set_joint_positions", done=False, extra={"object_id": args.object_id})
            return {
                "task": "set_joint_positions",
                "status": "success",
                "object_id": args.object_id,
                "joints_set": n,
                "controllable_joints": len(controllable),
                "message": f"Set {n} joint targets. Call step_simulation to animate.",
            }
    except Exception as e:
        raise RuntimeError(f"set_joint_positions failed: {e}") from e


# ======================
# Tool: Move End Effector (IK-based)
# ======================
class MoveEndEffectorArgs(BaseModel):
    object_id: int = Field(description="Body ID of the robot (from load_urdf).")
    target_position: list[float] = Field(
        description="Target [x, y, z] position for the end effector in world coordinates.",
    )
    end_effector_index: int = Field(
        default=-1,
        description="Link index of end effector. -1 = last link (auto-detect).",
    )
    max_force: float = Field(default=500.0, description="Motor force per joint.")


@mcp_server.tool()
def move_end_effector(args: MoveEndEffectorArgs):
    """
    Move robot end effector to a target position using inverse kinematics.

    This computes the required joint angles automatically and applies position control.
    After calling this, use step_simulation(steps=200+) to let the arm actually move.

    Use this instead of set_joint_positions when you know WHERE you want the
    end effector to go but don't know the joint angles.

    Returns:
    - computed joint_positions
    - end_effector_link used
    """
    try:
        _validate_vector(args.target_position, "target_position", size=3, allow_negative=True, max_val=100.0)
    except ValueError as e:
        return _tool_error("move_end_effector", str(e), "validation")

    try:
        with with_simulation() as cid:
            if not _is_body_valid(cid, args.object_id):
                return _tool_error("move_end_effector", f"Object ID {args.object_id} not found.", "validation")

            num_joints = p.getNumJoints(args.object_id, physicsClientId=cid)
            # Auto-detect end effector: use last non-fixed link
            ee_index = args.end_effector_index
            if ee_index < 0:
                ee_index = num_joints - 1
                for i in range(num_joints - 1, -1, -1):
                    info = p.getJointInfo(args.object_id, i, physicsClientId=cid)
                    if info[2] != p.JOINT_FIXED:
                        ee_index = i
                        break

            # Compute IK
            joint_positions = p.calculateInverseKinematics(
                args.object_id,
                ee_index,
                args.target_position,
                physicsClientId=cid,
            )

            # Apply position control to all controllable joints
            controllable = []
            for i in range(num_joints):
                info = p.getJointInfo(args.object_id, i, physicsClientId=cid)
                if info[2] != p.JOINT_FIXED:
                    controllable.append(i)

            n = min(len(joint_positions), len(controllable))
            for i in range(n):
                p.setJointMotorControl2(
                    bodyUniqueId=args.object_id,
                    jointIndex=controllable[i],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=args.max_force,
                    physicsClientId=cid,
                )

            _publish_snapshot("move_end_effector", done=False, extra={"object_id": args.object_id})
            return {
                "task": "move_end_effector",
                "status": "success",
                "object_id": args.object_id,
                "end_effector_link": ee_index,
                "target_position": args.target_position,
                "joint_positions": [round(float(v), 4) for v in joint_positions[:n]],
                "message": f"IK solved. Call step_simulation(steps=200) to animate arm to target.",
            }
    except Exception as e:
        raise RuntimeError(f"move_end_effector failed: {e}") from e


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
        raise RuntimeError(f"delete_object failed: {e}") from e


# ======================
# Tool: Get Simulation Info
# ======================
@mcp_server.tool()
def get_simulation_info():
    """
    获取仿真全景信息，包含所有物体的 id/位置/姿态。

    返回：
    - timestep: 当前时间步
    - num_bodies: 物体数量（含 plane）
    - gravity: 重力设置
    - objects: 列表，每项 {id, position:[x,y,z], orientation:[x,y,z,w], num_joints}
      （跳过 plane，plane_id=0）
    """
    try:
        with with_simulation() as cid:
            params = p.getPhysicsEngineParameters(physicsClientId=cid)
            num = p.getNumBodies(physicsClientId=cid)
            objects = []
            for body_id in range(num):
                if body_id == _plane_body_id:
                    continue
                try:
                    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=cid)
                    num_joints = p.getNumJoints(body_id, physicsClientId=cid)
                    objects.append({
                        "id": body_id,
                        "position": [round(float(v), 4) for v in pos],
                        "orientation": [round(float(v), 4) for v in orn],
                        "num_joints": int(num_joints),
                    })
                except Exception:
                    continue
            return {
                "task": "get_simulation_info",
                "status": "success",
                "timestep": params.get('fixedTimeStep', 0.0),
                "num_bodies": num,
                "gravity": [
                    params.get('gravityAccelerationX', 0.0),
                    params.get('gravityAccelerationY', 0.0),
                    params.get('gravityAccelerationZ', -9.8),
                ],
                "objects": objects,
                "message": f"Scene has {len(objects)} object(s) (excluding plane).",
            }
    except Exception as e:
        raise RuntimeError(f"get_simulation_info failed: {e}") from e


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
        raise RuntimeError(f"set_gravity failed: {e}") from e


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
