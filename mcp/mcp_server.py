from fastmcp import FastMCP
import pybullet as p
import pybullet_data
import time
import math
import os
import tempfile

mcp_server = FastMCP("pybullet")


# ======================
# PyBullet 基础工具
# ======================
def setup_sim(gui=False):
    if gui:
        cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    return cid


def cleanup_sim():
    p.disconnect()


def step(n=240):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


# ======================
# Tool 1: stacking
# ======================
@mcp_server.tool()
def stacking():
    """
    Stack two cubes using PyBullet.
    """
    cid = setup_sim(gui=False)

    cube1 = p.loadURDF("cube_small.urdf", [0, 0, 0.02])
    cube2 = p.loadURDF("cube_small.urdf", [0, 0, 0.06])

    step(240)

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
@mcp_server.tool()
def grab_and_place():
    """
    Simulate grab and place (simplified, no gripper).
    """
    cid = setup_sim(gui=False)

    cube = p.loadURDF("cube_small.urdf", [0.2, 0, 0.02])

    # "Grab" = teleport
    p.resetBasePositionAndOrientation(cube, [0, 0.2, 0.2], [0, 0, 0, 1])
    step(120)

    # "Place"
    p.resetBasePositionAndOrientation(cube, [0.4, 0.4, 0.02], [0, 0, 0, 1])
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
@mcp_server.tool()
def path_tracking():
    """
    Track a circular path using a sphere.
    """
    cid = setup_sim(gui=False)

    sphere = p.loadURDF("sphere2.urdf", [0.3, 0, 0.1])

    path = []
    radius = 0.3

    for i in range(120):
        angle = 2 * math.pi * i / 120
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
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
# Run MCP Server
# ======================
if __name__ == "__main__":
    mcp_server.run(transport="stdio")
