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


# 路径规划工具
class PathPlanningArgs(BaseModel):
    start_position: list[float] = Field(
        default=[0.2, 0.0, 0.02], description="Start position of the arm."
    )
    target_position: list[float] = Field(
        default=[0.4, 0.4, 0.02], description="Target position."
    )
    steps: int = Field(default=240, description="Steps for planning and moving.")


@mcp_server.tool()
def path_planning(args: PathPlanningArgs):
    """
    Plan and move robot arm from start to target position.
    """
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=args.start_position)

    # 假设这里的路径规划为线性移动
    move_vector = [
        (args.target_position[0] - args.start_position[0]) / args.steps,
        (args.target_position[1] - args.start_position[1]) / args.steps,
        (args.target_position[2] - args.start_position[2]) / args.steps,
    ]

    for i in range(args.steps):
        new_pos = [
            args.start_position[0] + move_vector[0] * i,
            args.start_position[1] + move_vector[1] * i,
            args.start_position[2] + move_vector[2] * i,
        ]
        # Move the robot arm
        p.resetBasePositionAndOrientation(robot_id, new_pos, [0, 0, 0, 1])
        step(1)

    final_pos, _ = p.getBasePositionAndOrientation(robot_id)

    return {
        "task": "path_planning",
        "status": "success",
        "final_position": final_pos,
        "message": "Robot arm moved to target position.",
    }


# 增加摩擦力和弹性
class FrictionAndElasticityArgs(BaseModel):
    friction: float = Field(default=0.5, description="Friction coefficient.")
    restitution: float = Field(default=0.9, description="Elasticity (bounciness).")


@mcp_server.tool()
def adjust_physics(args: FrictionAndElasticityArgs):
    """
    Adjust friction and elasticity of the environment objects.
    """
    cube = p.loadURDF("cube_small.urdf", [0, 0, 0.02])

    # 调整摩擦力和弹性
    p.changeDynamics(
        cube, -1, lateralFriction=args.friction, restitution=args.restitution
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
        description="List of initial object positions.",
    )
    target_position: list[float] = Field(
        default=[0.6, 0.6, 0.02], description="Target position to place objects."
    )


@mcp_server.tool()
def multi_object_grab_and_place(args: MultiObjectGrabArgs):
    """
    Grab and place multiple objects at the same time.
    """
    cubes = []
    for pos in args.object_positions:
        cubes.append(p.loadURDF("cube_small.urdf", pos))

    # 抓取并移动到目标位置
    for cube in cubes:
        p.resetBasePositionAndOrientation(cube, args.target_position, [0, 0, 0, 1])

    return {
        "task": "multi_object_grab_and_place",
        "status": "success",
        "message": "Multiple objects moved to the target position.",
    }


# 模拟视觉传感器
class VisionSensorArgs(BaseModel):
    width: int = Field(default=640, description="Image width of the camera.")
    height: int = Field(default=480, description="Image height of the camera.")
    view_matrix: list[float] = Field(
        default=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], description="Camera view matrix."
    )
    projection_matrix: list[float] = Field(
        default=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        description="Camera projection matrix.",
    )


@mcp_server.tool()
def simulate_vision_sensor(args: VisionSensorArgs):
    """
    Simulate a vision sensor to capture an image from the scene.
    """
    width, height = args.width, args.height
    img_arr = p.getCameraImage(
        width,
        height,
        viewMatrix=args.view_matrix,
        projectionMatrix=args.projection_matrix,
    )
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
    Cleanup the PyBullet simulation environment once all tasks are finished.
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
    Check the current state of the simulation and return object data.
    """
    num_objects = p.getNumBodies()  # 获取场景中的物体数量
    state_data = {}

    # 获取所有物体的状态
    for obj_id in range(num_objects):
        pos, _ = p.getBasePositionAndOrientation(
            obj_id
        )  # 获取物体的唯一 ID 并获取其位置和姿态
        state_data[obj_id] = pos  # 以物体 ID 为键，将位置存储在字典中

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
