# Docker Guide

This folder contains service containers used by RobotAgent development and simulation.

- `pybullet/` builds and runs the PyBullet MCP service.
- `gazebo/` builds and runs the Gazebo MCP service.
- `qdrant/` is expected by the README as the local vector database compose target when present.

Keep container ports aligned with `config/config.yml` and backend health checks. Avoid baking personal absolute paths or secrets into Dockerfiles or compose files; prefer environment variables and mounted project directories.
