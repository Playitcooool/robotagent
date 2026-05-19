# Tests Guide

This folder contains Python tests for tools, intent workflow, MCP interactions, navigation evaluation, and simulation streaming.

Use the repo uv environment:

```bash
rtk .venv/bin/pytest tests
```

Prefer targeted tests while iterating, then run the broader relevant subset before finishing. Tests may require local services such as Redis, Qdrant, or MCP servers depending on the module under test; call out missing service dependencies in final notes if they prevent verification.
