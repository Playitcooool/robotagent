import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("pybullet")


@pytest.fixture(scope="module")
def mcp_module():
    spec = importlib.util.spec_from_file_location("mcp_server_mobile_tests", ROOT / "mcp" / "mcp_server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_mobile_tests"] = module
    spec.loader.exec_module(module)
    yield module
    try:
        module.cleanup_simulation()
    except Exception:
        pass


def test_load_robot_rejects_unknown_type(mcp_module):
    result = mcp_module.load_robot(mcp_module.LoadRobotArgs(robot_type="walker"))

    assert result["status"] == "error"
    assert result["error_type"] == "validation"
    assert "Unsupported robot_type" in result["message"]


def test_load_robot_requires_custom_urdf_path(mcp_module):
    result = mcp_module.load_robot(mcp_module.LoadRobotArgs(robot_type="custom_urdf", urdf_path=""))

    assert result["status"] == "error"
    assert result["error_type"] == "validation"
    assert "urdf_path is required" in result["message"]


def test_configure_robot_drive_rejects_bad_drive_type(mcp_module):
    result = mcp_module.configure_robot_drive(
        mcp_module.ConfigureRobotDriveArgs(robot_id=1, drive_type="skid_magic")
    )

    assert result["status"] == "error"
    assert result["error_type"] == "validation"
    assert "Unsupported drive_type" in result["message"]


def test_apply_robot_drive_velocity_base_sets_body_velocity(mcp_module):
    pybullet = pytest.importorskip("pybullet")

    cid = pybullet.connect(pybullet.DIRECT)
    try:
        body_id = pybullet.createMultiBody(baseMass=1.0, basePosition=[0, 0, 0], physicsClientId=cid)
        mcp_module._robot_drive_registry[body_id] = {
            "drive_type": "velocity_base",
            "wheel_joints": [],
            "wheel_radius": 0.1,
            "wheel_base": 0.4,
        }

        mode = mcp_module._apply_robot_drive(cid, body_id, 0.5, 0.25, 10.0)
        lin_vel, ang_vel = pybullet.getBaseVelocity(body_id, physicsClientId=cid)

        assert mode == "velocity_base"
        assert lin_vel[0] == pytest.approx(0.5)
        assert lin_vel[1] == pytest.approx(0.0)
        assert ang_vel[2] == pytest.approx(0.25)
    finally:
        pybullet.disconnect(cid)


def test_run_pybullet_navigation_task_rejects_bad_waypoints(mcp_module):
    result = mcp_module.run_pybullet_navigation_task(
        mcp_module.RunPybulletNavigationTaskArgs(waypoints=[])
    )

    assert result["status"] == "error"
    assert result["error_type"] == "validation"


def test_run_pybullet_navigation_task_returns_metrics_schema(mcp_module):
    mcp_module.cleanup_simulation()
    result = mcp_module.run_pybullet_navigation_task(
        mcp_module.RunPybulletNavigationTaskArgs(
            robot_type="r2d2",
            start_position=[0.0, 0.0, 0.0],
            waypoints=[[0.05, 0.0]],
            speed=0.5,
            tolerance=0.2,
            max_steps=120,
            publish_frames=False,
        )
    )

    assert result["backend"] == "pybullet"
    assert result["status"] in {"success", "warning"}
    assert set(result["metrics"]) >= {
        "completed",
        "success",
        "final_error",
        "waypoint_errors",
        "path_length",
        "steps",
        "collision_count",
        "min_clearance",
        "failure_reason",
    }
    state = mcp_module.get_robot_state(mcp_module.GetRobotStateArgs(robot_id=result["robot_id"], include_joints=False))
    assert state["base_linear_velocity"][0] == pytest.approx(0.0)
    assert state["base_linear_velocity"][1] == pytest.approx(0.0)
