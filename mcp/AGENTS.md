# MCP Guide

This folder contains simulation MCP servers, reusable simulation assets, and evaluation helpers.

- `mcp_server.py` is the PyBullet MCP server. It owns simulation state, robot/object operations, camera/frame capture, and writes shared frames into `.sim_stream`.
- `gazebo_mcp_server.py` exposes Gazebo-oriented simulation tools.
- `navigation_eval.py` validates and scores mobile robot navigation paths.
- `assets/pybullet_models/` and `assets/gazebo_models/` contain robot and object models used by the servers.
- `.sim_stream/` is a runtime frame/replay directory consumed by `backend/routes_sim.py`; treat it as generated state.

Simulation tools are called by agents, so prefer explicit validation, bounded step counts, and structured error payloads over exceptions that leak implementation details.

When changing stream output paths or metadata shape, update backend simulation routes and any frontend simulation display code together.
