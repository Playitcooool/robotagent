# RobotAgent Frontend (Vue 3)

A small, clean Vue 3 frontend for the RobotAgent project.

## Layout
- Left: conversation list (history)
- Middle: chat / composer (send messages to model)
- Right: tool results (render text, JSON, images)

## Quick start
1. cd frontend
2. npm install
3. npm run dev

By default the UI calls these endpoints (expected on your backend):
- `GET /api/messages` → returns recent messages array
- `POST /api/chat/send` with { message } → returns { reply, tool_result? }

If those endpoints are not present, the UI has graceful fallbacks and will show local mocked responses.

## Notes
- This is intentionally minimal. If you want, I can integrate Pinia for global state, add WebSocket streaming, or wire real endpoints to your agent.
- No modifications were made to existing backend code (except adding a small `server.py` that exposes a lightweight `/api/chat/send` shim you can run to try chat locally).

## Running the simple Python API server
A minimal FastAPI server `server.py` is provided to accept chat messages and forward them to your existing `agent` (from `main.py`).

Install runtime deps (if not already installed):

```bash
pip install fastapi uvicorn
```

Run the server (from project root):

```bash
uvicorn server:app --reload --port 8000
```

The frontend expects POST `/api/chat/send` with JSON `{ "message": "..." }`, and will read the JSON response `{ "reply": "..." }`.

If `main.agent` is unavailable, the server will return a simple echo-style simulated reply to allow frontend testing.
