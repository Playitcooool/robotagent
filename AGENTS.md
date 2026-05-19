@/Users/weiciruan/.codex/RTK.md

# RobotAgent Repo Guide

RobotAgent is a multi-agent robotics task execution platform. The main product path is:

- FastAPI backend in `backend/`
- LangChain/LangGraph tools in `tools/`
- Prompt assembly in `prompts/`
- MCP simulation services in `mcp/`
- Vue 3 workbench frontend in `frontend/`
- RAG indexing/query scripts in `RAG/`
- evaluation and training-free GRPO experiments in `experiments/` and `training_free_grpo/`

The runtime configuration lives in `config/config.yml`. Do not hard-code model names, MCP URLs, Redis URLs, or API keys in code or docs when the config file or environment variables already own them.

# Command Environment

Use the existing uv environment in `.venv` for Python commands. Prefer:

```bash
rtk .venv/bin/python ...
rtk .venv/bin/pytest ...
```

For frontend work, run commands from `frontend/` unless intentionally touching the root package files:

```bash
rtk npm run build
```

# Big Picture

- `server.py` starts the FastAPI app exposed from `backend/app.py`.
- `backend/app.py` wires auth, chat/session APIs, streaming responses, Redis checkpoint/history storage, model config, and tool loading.
- `tools/GeneralTool.py`, `tools/AnalysisTool.py`, and `tools/SubAgentTool.py` are the main tool surface used by the agents.
- `prompts/*Prompt.py` controls the main, analysis, and simulation agent behavior.
- `mcp/mcp_server.py` and `mcp/gazebo_mcp_server.py` expose simulation capabilities for PyBullet and Gazebo.
- `frontend/src/components/ChatView.vue` and the `frontend/src/components/chat/` files drive the chat UI; `frontend/src/views/WorkbenchView.vue` frames the workbench.
- `RAG/script/` builds local knowledge collections; `tools/GeneralTool.py` queries them through Qdrant when enabled.

# Development Notes

- Keep generated artifacts out of runtime paths unless the task is explicitly about documents, charts, or output files.
- Use `README.md` for user-facing setup details and these `AGENTS.md` files for Codex-facing navigation and guardrails.
- Existing tests are mostly in `tests/` with some legacy RAG checks under `RAG/`.
- Redis is a required backend dependency: DB 0 for LangGraph checkpoints, DB 1 for chat history, DB 2 for auth/session state.
- Treat `documents/`, `output/`, and `charts/figures/` as artifact-heavy areas. Avoid broad rewrites there unless requested.
