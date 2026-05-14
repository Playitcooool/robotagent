"""Shared navigation task validation and metrics helpers."""

from __future__ import annotations

import math
from typing import Any


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    return number


def validate_waypoints(waypoints: list[list[float]]) -> list[list[float]]:
    """Return normalized planar waypoints as [[x, y], ...]."""
    if not isinstance(waypoints, (list, tuple)) or not waypoints:
        raise ValueError("waypoints must be a non-empty list")
    normalized: list[list[float]] = []
    for idx, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
            raise ValueError(f"waypoint {idx} must contain at least x and y")
        normalized.append([
            _finite_number(waypoint[0], f"waypoint {idx} x"),
            _finite_number(waypoint[1], f"waypoint {idx} y"),
        ])
    return normalized


def normalize_start_position(position: list[float]) -> list[float]:
    if not isinstance(position, (list, tuple)):
        raise ValueError("start_position must be a list")
    if len(position) not in (2, 3):
        raise ValueError("start_position must have 2 or 3 values")
    values = [_finite_number(v, f"start_position[{i}]") for i, v in enumerate(position)]
    if len(values) == 2:
        values.append(0.0)
    return values


def normalize_obstacles(obstacles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize simple box/sphere/cylinder obstacle specs for both backends."""
    if obstacles is None:
        return []
    if not isinstance(obstacles, (list, tuple)):
        raise ValueError("obstacles must be a list")
    normalized: list[dict[str, Any]] = []
    for idx, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict):
            raise ValueError(f"obstacle {idx} must be an object")
        shape = str(obstacle.get("shape", obstacle.get("type", "box"))).lower()
        if shape not in {"box", "sphere", "cylinder"}:
            raise ValueError(f"obstacle {idx} shape must be box, sphere, or cylinder")
        position = obstacle.get("position", [0.0, 0.0, 0.5])
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"obstacle {idx} position must have 3 values")
        size = obstacle.get("size", [0.2, 0.2, 0.2])
        if not isinstance(size, (list, tuple)) or not size:
            raise ValueError(f"obstacle {idx} size must be a non-empty list")
        pos = [_finite_number(v, f"obstacle {idx} position[{i}]") for i, v in enumerate(position)]
        raw_size = [_finite_number(v, f"obstacle {idx} size[{i}]") for i, v in enumerate(size)]
        if any(v <= 0 for v in raw_size):
            raise ValueError(f"obstacle {idx} size values must be positive")
        if shape == "box":
            if len(raw_size) == 1:
                size_values = raw_size * 3
            elif len(raw_size) == 2:
                size_values = [raw_size[0], raw_size[1], raw_size[1]]
            else:
                size_values = raw_size[:3]
        elif shape == "sphere":
            size_values = [raw_size[0]]
        else:
            size_values = [raw_size[0], raw_size[1] if len(raw_size) > 1 else raw_size[0]]
        normalized.append({
            "name": str(obstacle.get("name", f"nav_obstacle_{idx}")),
            "shape": shape,
            "position": pos,
            "size": size_values,
        })
    return normalized


def path_length(trajectory: list[list[float]]) -> float:
    total = 0.0
    for a, b in zip(trajectory, trajectory[1:]):
        total += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return total


def waypoint_errors(final_position: list[float], waypoints: list[list[float]]) -> list[dict[str, Any]]:
    return [
        {
            "waypoint": [float(wp[0]), float(wp[1])],
            "error": math.hypot(float(wp[0]) - float(final_position[0]), float(wp[1]) - float(final_position[1])),
        }
        for wp in waypoints
    ]


def obstacle_clearance(
    position: list[float],
    obstacle: dict[str, Any],
    *,
    robot_radius: float = 0.25,
) -> float:
    px, py = float(position[0]), float(position[1])
    ox, oy = float(obstacle["position"][0]), float(obstacle["position"][1])
    shape = obstacle["shape"]
    size = obstacle["size"]
    if shape == "box":
        hx = float(size[0]) / 2.0
        hy = float(size[1]) / 2.0
        dx = max(abs(px - ox) - hx, 0.0)
        dy = max(abs(py - oy) - hy, 0.0)
        distance = math.hypot(dx, dy)
    else:
        distance = math.hypot(px - ox, py - oy) - float(size[0])
    return distance - robot_radius


def approximate_collisions(
    trajectory: list[list[float]],
    obstacles: list[dict[str, Any]],
    *,
    robot_radius: float = 0.25,
) -> tuple[int, float | None]:
    if not trajectory or not obstacles:
        return 0, None
    collision_count = 0
    min_clearance: float | None = None
    for point in trajectory:
        for obstacle in obstacles:
            clearance = obstacle_clearance(point, obstacle, robot_radius=robot_radius)
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            if clearance <= 0.0:
                collision_count += 1
    return collision_count, min_clearance


def build_navigation_result(
    *,
    backend: str,
    task: str,
    robot_id: int | str | None,
    start_position: list[float],
    waypoints: list[list[float]],
    trajectory: list[list[float]],
    steps: int,
    tolerance: float,
    collision_count: int = 0,
    min_clearance: float | None = None,
    status: str | None = None,
    failure_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_position = trajectory[-1] if trajectory else start_position
    final_error = math.hypot(float(waypoints[-1][0]) - float(final_position[0]), float(waypoints[-1][1]) - float(final_position[1]))
    completed = final_error <= float(tolerance)
    success = completed and collision_count == 0 and not failure_reason
    reason = failure_reason
    if not reason and not completed:
        reason = "max_steps_exceeded"
    if not reason and collision_count > 0:
        reason = "collision_detected"
    payload = {
        "task": task,
        "status": status or ("success" if success else "warning"),
        "backend": backend,
        "robot_id": robot_id,
        "start_position": start_position,
        "waypoints": waypoints,
        "trajectory": trajectory,
        "final_position": final_position,
        "metrics": {
            "completed": completed,
            "success": success,
            "final_error": final_error,
            "waypoint_errors": waypoint_errors(final_position, waypoints),
            "path_length": path_length(trajectory),
            "steps": int(steps),
            "collision_count": int(collision_count),
            "min_clearance": min_clearance,
            "failure_reason": reason,
        },
    }
    if extra:
        payload.update(extra)
    return payload
