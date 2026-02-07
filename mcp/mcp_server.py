import asyncio
import pybullet as p
import pybullet_data
import time
import math
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


def setup_simulation(gui: bool = False):
    """初始化PyBullet环境，只在首次创建时运行"""
    global simulation_instance
    if simulation_instance is None:
        if gui:
            simulation_instance = p.connect(p.GUI)
        else:
            simulation_instance = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
    else:
        print("PyBullet environment is already running.")


def cleanup_simulation():
    """关闭PyBullet环境"""
    global simulation_instance
    if simulation_instance:
        p.disconnect()
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
    gui: bool = Field(default=False, description="Whether to enable PyBullet GUI.")


@mcp_server.tool()
def initialize_simulation(args: InitializeSimulationArgs):
    """
    Initialize PyBullet simulation environment and keep it running until the program ends.
    """
    setup_simulation(gui=args.gui)

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
        default=[0.0, 0.0, 0.02], description="Initial position of the cube."
    )
    push_vector: list[float] = Field(
        default=[0.2, 0.0, 0.0], description="Vector along which to push the cube."
    )
    steps: int = Field(default=120, description="Number of simulation steps.")


@mcp_server.tool()
def push_cube_step(args: PushCubeStepArgs):
    """
    Push a cube step-by-step in the simulation environment.
    """
    cube = p.loadURDF("cube_small.urdf", args.start_position)

    # 线性插值推动
    start_pos = args.start_position
    dx = args.push_vector[0] / args.steps
    dy = args.push_vector[1] / args.steps
    dz = args.push_vector[2] / args.steps

    # 每次调用执行一个小步进
    for i in range(args.steps):
        new_pos = [
            start_pos[0] + dx * i,
            start_pos[1] + dy * i,
            start_pos[2] + dz * i,
        ]
        p.resetBasePositionAndOrientation(cube, new_pos, [0, 0, 0, 1])
        step(1)

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    return {
        "task": "push_cube_step",
        "status": "success",
        "final_position": final_pos,
        "message": f"Cube pushed along vector {args.push_vector}.",
    }


# ======================
# Tool 3: Grab and Place Step-by-Step
# ======================
class GrabAndPlaceStepArgs(BaseModel):
    start_position: list[float] = Field(
        default=[0.2, 0.0, 0.02], description="Initial position of the object."
    )
    target_position: list[float] = Field(
        default=[0.4, 0.4, 0.02], description="Target position to place the object."
    )
    steps: int = Field(default=120, description="Number of steps to simulate.")


@mcp_server.tool()
def grab_and_place_step(args: GrabAndPlaceStepArgs):
    """
    Grab and place object step-by-step in the simulation environment.
    """
    cube = p.loadURDF("cube_small.urdf", args.start_position)

    # Lift object (teleport)
    p.resetBasePositionAndOrientation(cube, [0, 0.2, 0.2], [0, 0, 0, 1])
    step(60)

    # Place object at the target position
    p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1])
    step(args.steps)

    final_pos, _ = p.getBasePositionAndOrientation(cube)

    return {
        "task": "grab_and_place_step",
        "status": "success",
        "final_position": final_pos,
        "message": f"Object placed at target location {args.target_position}.",
    }


# ======================
# Tool 4: Cleanup Simulation
# ======================
@mcp_server.tool()
def cleanup_simulation_tool():
    """
    Cleanup the PyBullet simulation environment once all tasks are finished.
    """
    cleanup_simulation()
    return {
        "task": "cleanup_simulation",
        "status": "success",
        "message": "Simulation environment cleaned up.",
    }


# ======================
# 启动 MCP 服务
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
