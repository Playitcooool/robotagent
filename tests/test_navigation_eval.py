import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("navigation_eval_tests", ROOT / "mcp" / "navigation_eval.py")
navigation_eval = importlib.util.module_from_spec(spec)
sys.modules["navigation_eval_tests"] = navigation_eval
spec.loader.exec_module(navigation_eval)

approximate_collisions = navigation_eval.approximate_collisions
build_navigation_result = navigation_eval.build_navigation_result
normalize_obstacles = navigation_eval.normalize_obstacles
path_length = navigation_eval.path_length
validate_waypoints = navigation_eval.validate_waypoints


def test_validate_waypoints_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_waypoints([])


def test_path_length_uses_planar_distance():
    assert path_length([[0, 0, 0], [3, 4, 0], [6, 8, 2]]) == pytest.approx(10.0)


def test_collision_approximation_for_box():
    obstacles = normalize_obstacles([
        {"shape": "box", "position": [1.0, 0.0, 0.5], "size": [0.5, 0.5, 1.0]}
    ])

    collisions, min_clearance = approximate_collisions([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], obstacles)

    assert collisions == 1
    assert min_clearance < 0.0


def test_navigation_result_schema_success():
    result = build_navigation_result(
        backend="test",
        task="run_navigation",
        robot_id=7,
        start_position=[0.0, 0.0, 0.0],
        waypoints=[[1.0, 0.0]],
        trajectory=[[0.0, 0.0, 0.0], [1.02, 0.0, 0.0]],
        steps=3,
        tolerance=0.05,
    )

    metrics = result["metrics"]
    assert metrics["completed"] is True
    assert metrics["success"] is True
    assert metrics["final_error"] == pytest.approx(0.02)
    assert metrics["path_length"] == pytest.approx(1.02)
    assert metrics["failure_reason"] is None
