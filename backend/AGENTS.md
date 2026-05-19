# Backend Guide

This folder contains the FastAPI application and server-side workflow glue.

- `app.py` is the main application entry: startup/shutdown, Redis clients, agent initialization, CORS, health/tools endpoints, chat streaming, and session persistence.
- `routes_auth.py`, `auth_utils.py`, and `schemas.py` handle registration, login, password validation/hashing, sessions, and request models.
- `routes_sim.py` exposes simulation frame/debug endpoints that read from `mcp/.sim_stream`.
- `model_config.py` resolves OpenAI-compatible model settings from `config/config.yml` and request overrides.
- `stream_utils.py` normalizes LangChain/LangGraph stream events into frontend-friendly deltas.
- `workflow_utils.py` contains small workflow/environment helpers.
- `utils/retry.py` provides async retry behavior used during service startup.

Use `.venv` for Python commands from the repo root, for example `rtk .venv/bin/pytest tests/test_intent_workflow.py`. Keep Redis DB separation intact: checkpoints, chat history, and auth state intentionally use different URLs.

When changing chat behavior, inspect `frontend/src/composables/useSSE.js` and chat components too, because stream payload shape is a cross-layer contract.
