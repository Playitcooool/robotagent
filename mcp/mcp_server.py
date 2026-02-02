import asyncio
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import pybullet as p
import pybullet_data
import time
import math

mcp_server = FastMCP("pybullet")


# ======================
# PyBullet 基础工具（不暴露给 LLM）
# ======================
def setup_sim(gui: bool = False):
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")


def cleanup_sim():
    p.disconnect()


def step(n: int = 240):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


# ======================
# Tool 1: stacking
# ======================
class StackingArgs(BaseModel):
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")
    settle_steps: int = Field(
        default=240, description="Simulation steps to let cubes settle."
    )


@mcp_server.tool()
def stacking(args: StackingArgs):
    """
    Stack two cubes using PyBullet.
    """
    setup_sim(gui=args.gui)

    cube1 = p.loadURDF("cube_small.urdf", [0, 0, 0.02])
    cube2 = p.loadURDF("cube_small.urdf", [0, 0, 0.06])

    step(args.settle_steps)

    pos1, _ = p.getBasePositionAndOrientation(cube1)
    pos2, _ = p.getBasePositionAndOrientation(cube2)

    cleanup_sim()

    return {
        "task": "stacking",
        "status": "success",
        "cube1_position": pos1,
        "cube2_position": pos2,
        "message": "Two cubes stacked successfully.",
    }


# ======================
# Tool 2: grab_and_place
# ======================
class GrabAndPlaceArgs(BaseModel):
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")
    start_position: list[float] = Field(
        default=[0.2, 0.0, 0.02], description="Initial position of the object."
    )
    target_position: list[float] = Field(
        default=[0.4, 0.4, 0.02], description="Target position to place the object."
    )


@mcp_server.tool()
def grab_and_place(args: GrabAndPlaceArgs):
    """
    Simulate grab and place (simplified, teleport-based).
    """
    setup_sim(gui=args.gui)

    cube = p.loadURDF("cube_small.urdf", args.start_position)

    # Grab (teleport upward)
    p.resetBasePositionAndOrientation(cube, [0, 0.2, 0.2], [0, 0, 0, 1])
    step(120)

    # Place
    p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1])
    step(240)

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    cleanup_sim()

    return {
        "task": "grab_and_place",
        "status": "success",
        "final_position": final_pos,
        "message": "Object grabbed and placed at target location.",
    }


# ======================
# Tool 3: path_tracking
# ======================
class PathTrackingArgs(BaseModel):
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")
    radius: float = Field(default=0.3, description="Radius of the circular path.")
    steps: int = Field(default=120, description="Number of trajectory points.")


@mcp_server.tool()
def path_tracking(args: PathTrackingArgs):
    """
    Track a circular path using a sphere.
    """
    setup_sim(gui=args.gui)

    sphere = p.loadURDF("sphere2.urdf", [args.radius, 0, 0.1])

    path = []

    for i in range(args.steps):
        angle = 2 * math.pi * i / args.steps
        x = args.radius * math.cos(angle)
        y = args.radius * math.sin(angle)
        z = 0.1
        p.resetBasePositionAndOrientation(sphere, [x, y, z], [0, 0, 0, 1])
        step(2)
        path.append((x, y, z))

    cleanup_sim()

    return {
        "task": "path_tracking",
        "status": "success",
        "trajectory_points": len(path),
        "message": "Circular path tracked successfully.",
    }


# ======================
# Tool 4: push_cube
# ======================
class PushCubeArgs(BaseModel):
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")
    start_position: list[float] = Field(
        default=[0.0, 0.0, 0.02], description="Initial position of the cube."
    )
    push_vector: list[float] = Field(
        default=[0.2, 0.0, 0.0], description="Vector along which to push the cube."
    )
    steps: int = Field(default=120, description="Number of simulation steps.")


@mcp_server.tool()
def push_cube(args: PushCubeArgs):
    """
    Push a cube along a given vector using PyBullet.
    """
    setup_sim(gui=args.gui)

    cube = p.loadURDF("cube_small.urdf", args.start_position)

    # 线性插值推动
    start_pos = args.start_position
    dx = args.push_vector[0] / args.steps
    dy = args.push_vector[1] / args.steps
    dz = args.push_vector[2] / args.steps

    for i in range(args.steps):
        new_pos = [start_pos[0] + dx * i, start_pos[1] + dy * i, start_pos[2] + dz * i]
        p.resetBasePositionAndOrientation(cube, new_pos, [0, 0, 0, 1])
        step(1)

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    cleanup_sim()

    return {
        "task": "push_cube",
        "status": "success",
        "final_position": final_pos,
        "message": f"Cube pushed along vector {args.push_vector}.",
    }


# ======================
# Tool 5: pick_and_throw
# ======================
class PickAndThrowArgs(BaseModel):
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")
    start_position: list[float] = Field(
        default=[0.0, 0.0, 0.02], description="Initial position of the cube."
    )
    throw_vector: list[float] = Field(
        default=[0.3, 0.3, 0.2], description="Vector to throw the cube."
    )
    settle_steps: int = Field(
        default=120, description="Steps to let cube settle before throw."
    )


@mcp_server.tool()
def pick_and_throw(args: PickAndThrowArgs):
    """
    Grab a cube, lift it, and throw it to a target position (teleport-based).
    """
    setup_sim(gui=args.gui)

    cube = p.loadURDF("cube_small.urdf", args.start_position)

    # Settle
    step(args.settle_steps)

    # Lift (grab)
    lift_pos = [
        args.start_position[0],
        args.start_position[1],
        args.start_position[2] + 0.2,
    ]
    p.resetBasePositionAndOrientation(cube, lift_pos, [0, 0, 0, 1])
    step(60)

    # Throw (teleport)
    throw_pos = [
        lift_pos[0] + args.throw_vector[0],
        lift_pos[1] + args.throw_vector[1],
        lift_pos[2] + args.throw_vector[2],
    ]
    p.resetBasePositionAndOrientation(cube, throw_pos, [0, 0, 0, 1])
    step(120)

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    cleanup_sim()

    return {
        "task": "pick_and_throw",
        "status": "success",
        "final_position": final_pos,
        "message": f"Cube picked and thrown along {args.throw_vector}.",
    }


# ======================
# 适配异步启动的核心逻辑（关键修改）
# ======================
async def start_mcp_server():
    """显式定义异步启动函数，确保绑定 0.0.0.0"""
    # 调用异步启动方法，明确指定 host 和 port
    await mcp_server.run_http_async(
        host="0.0.0.0",  # 强制绑定所有网卡，外部可访问
        port=8000,  # 容器内端口
        transport="http",
    )


# ======================
# Run MCP Server
# ======================
if __name__ == "__main__":
    # 显式运行异步事件循环，避免隐式配置问题
    # 启动异步服务并阻塞，直到服务停止
    asyncio.run(start_mcp_server())
