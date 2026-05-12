import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("pybullet")
pytest.importorskip("fastmcp")
pytest.importorskip("pydantic")


def load_mcp_module():
    module_path = Path(__file__).resolve().parents[1] / "mcp" / "mcp_server.py"
    spec = importlib.util.spec_from_file_location("mcp_server_streaming_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step_simulation_caps_preview_frames_and_publishes_final(monkeypatch):
    m = load_mcp_module()
    m.simulation_instance = 123
    stepped_batches = []
    published = []

    monkeypatch.setattr(m, "ensure_simulation", lambda: None)
    monkeypatch.setattr(m, "_safe_step", lambda cid, steps: stepped_batches.append(steps) or steps)
    monkeypatch.setattr(
        m,
        "_publish_realtime_frame",
        lambda **kwargs: published.append(kwargs),
    )

    result = m.step_simulation(m.StepSimulationArgs(steps=300, max_preview_frames=12))

    assert result["status"] == "success"
    assert result["published_frames"] == 12
    assert len(published) == 12
    assert published[-1]["step_idx"] == 300
    assert published[-1]["done"] is True
    assert sum(stepped_batches) == 300
    assert all(frame["width"] == 640 and frame["height"] == 480 for frame in published)


def test_step_simulation_can_skip_preview_frames(monkeypatch):
    m = load_mcp_module()
    m.simulation_instance = 123
    stepped_batches = []
    published = []

    monkeypatch.setattr(m, "ensure_simulation", lambda: None)
    monkeypatch.setattr(m, "_safe_step", lambda cid, steps: stepped_batches.append(steps) or steps)
    monkeypatch.setattr(m, "_publish_realtime_frame", lambda **kwargs: published.append(kwargs))

    result = m.step_simulation(m.StepSimulationArgs(steps=300, publish_frames=False))

    assert result["status"] == "success"
    assert result["published_frames"] == 0
    assert stepped_batches == [300]
    assert published == []


def test_scene_bounds_cache_recomputes_only_after_invalidation(monkeypatch):
    m = load_mcp_module()
    calls = []
    aabb = ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    monkeypatch.setattr(m, "_scene_aabb", lambda cid: calls.append(cid) or aabb)

    assert m._cached_scene_bounds_payload(7)["center"] == [0.5, 1.0, 1.5]
    assert m._cached_scene_bounds_payload(7)["center"] == [0.5, 1.0, 1.5]
    assert calls == [7]

    m._invalidate_scene_bounds()
    assert m._cached_scene_bounds_payload(7)["center"] == [0.5, 1.0, 1.5]
    assert calls == [7, 7]
