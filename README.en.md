# RobotAgent

English | [简体中文](README.zh-CN.md)

RobotAgent is a multi-agent platform for robotics task execution. It combines chat interaction, sub-agent collaboration, MCP-based simulation control, Agentic RAG, and a real-time visual workbench.

## Preview

![RobotAgent workbench preview](screenshots/22d6fdafc39c72568b5fcf74bea97327.png)

![RobotAgent chat and task execution preview](screenshots/6cc1fb3678ae4d068ad68cad244ad235.png)

![RobotAgent simulation and analysis preview](screenshots/89061257528821dbb3f4b3759a59f1af.png)

## Overview

RobotAgent provides an end-to-end workflow for robotics tasks:

- The FastAPI backend handles auth, sessions, streaming chat, and simulation-frame APIs
- The main agent interprets tasks, calls tools, and delegates to sub-agents
- Sub-agents handle simulation execution and data analysis
- Simulation capabilities are exposed through MCP services for PyBullet and Gazebo
- The Vue 3 + Vite frontend displays chat, plans, tool results, and simulation frames
- The RAG module supports academic or technical retrieval and local knowledge-base queries

Model settings, MCP endpoints, and external service credentials are owned by [config/config.yml](config/config.yml). Avoid hard-coding model names, deployment URLs, or API keys in documentation or code when configuration already owns them.

## Repository Layout

The main runtime paths are:

```text
robotagent/
├── backend/                 # FastAPI app, auth, sessions, streaming chat, simulation APIs
├── frontend/                # Vue 3 frontend
├── tools/                   # Main-agent tools and analysis tools
├── prompts/                 # Main, analysis, and simulation agent prompts
├── mcp/                     # MCP services and simulation resources
├── RAG/                     # Retrieval and indexing scripts
├── training_free_grpo/      # Experience collection and training-free optimization scripts
├── tests/                   # Unit tests
├── config/config.yml        # Model, MCP, and external service configuration
├── server.py                # Uvicorn entry point
├── dev.sh                   # Development script that starts backend and frontend
└── requirements.txt         # Python dependencies
```

Notes:

- The repository root also contains papers, defense materials, generated outputs, and temporary resources. They are not required to run the platform.
- Frontend dependencies are defined in [frontend/package.json](frontend/package.json), not the root [package.json](package.json).

## Core Capabilities

- Multi-agent collaboration: main agent + `data-analyzer` + `simulator`
- Streaming chat: `/api/chat/send` returns incremental NDJSON messages
- Session management: Redis stores chat history, session indexes, and login state
- Real-time simulation frames: `/api/sim/stream` and `/api/sim/latest.png`
- Tool-based retrieval: workspace search, web search, academic search, and local RAG
- Analysis tools: CSV summaries, descriptive statistics, and chart generation

## Tech Stack

- Backend: FastAPI, LangChain, LangGraph, Redis
- Frontend: Vue 3, Vite, Markdown-It, KaTeX, highlight.js
- Simulation: PyBullet MCP, Gazebo MCP
- Retrieval: Qdrant, sentence-transformers, Tavily, arXiv/OpenAlex

## Requirements

- Python 3.10+
- Node.js 18+
- Redis 6+
- An available OpenAI-compatible LLM service
- Optional: Docker for PyBullet, Gazebo, or Qdrant services

## Installation

### 1. Install Python Dependencies

Use the repository's existing `.venv` environment when available:

```bash
rtk .venv/bin/python -m pip install -r requirements.txt
```

You can also use your own virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Configuration

The main configuration file is [config/config.yml](config/config.yml).

Check these sections before running the platform:

- `llm` / `model_url`: main-agent model and inference endpoint
- `analysis_llm` / `simulation_llm`: sub-agent model configuration
- `mcp`: PyBullet and Gazebo MCP endpoints
- `tavily.api_key`: web search
- `judge`: external judge model used by evaluation experiments

Recommendations:

- Replace bundled or placeholder credentials with your own values
- Do not commit real API keys to public repositories

Redis uses these DBs by default:

- `redis://127.0.0.1:6379/0`: LangGraph checkpoints
- `redis://127.0.0.1:6379/1`: chat history
- `redis://127.0.0.1:6379/2`: auth and session state

## Running

### Start the Backend

```bash
rtk .venv/bin/python server.py
```

Default URL: `http://127.0.0.1:8000`

### Start the Frontend

```bash
cd frontend
rtk npm run dev
```

Default URL: `http://127.0.0.1:5173`

### Start Backend and Frontend Together

```bash
./dev.sh
```

[dev.sh](dev.sh) clears existing processes on ports `8000` and `5173`, then starts the backend and frontend together. Use `BACKEND_PYTHON=/path/to/python ./dev.sh` to choose the backend Python interpreter.

### Redis

Redis is required for session management and agent checkpoints.

```bash
# Start a local macOS/Linux Redis installation
redis-server --daemonize yes

# Or use Docker
docker run -d --name robotagent-redis -p 6379:6379 redis:7
```

Verify Redis:

```bash
redis-cli ping
# Expected: PONG
```

## Simulation Services

### PyBullet / Gazebo MCP

Run the Python services directly or use Docker Compose.

```bash
# Python
rtk .venv/bin/python mcp/mcp_server.py
rtk .venv/bin/python mcp/gazebo_mcp_server.py
```

```bash
# Docker
docker compose -f docker/pybullet/docker-compose.yml up -d --build
docker compose -f docker/gazebo/docker-compose.yml up -d --build
```

In the current codebase:

- Backend health checks probe PyBullet on port `18001` by default
- Backend health checks probe Gazebo on port `8002` by default
- Sub-agents read MCP service addresses from the `mcp` section in [config/config.yml](config/config.yml)

### Qdrant

Start Qdrant if you want to enable local knowledge-base retrieval:

```bash
docker compose -f docker/qdrant/docker-compose.yml up -d
```

Default port: `6333`.

## API Overview

### System

- `GET /api/health`
- `GET /api/ping`
- `GET /api/tools`

### Auth

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

### Chat and Sessions

- `GET /api/messages`
- `GET /api/sessions`
- `DELETE /api/sessions/{session_id}`
- `POST /api/chat/send`

### Simulation

- `GET /api/sim/debug`
- `GET /api/sim/latest-frame`
- `GET /api/sim/latest.png`
- `GET /api/sim/stream`

## Key Code Locations

- [backend/app.py](backend/app.py): app entry point, agent initialization, chat and session APIs
- [backend/routes_auth.py](backend/routes_auth.py): registration, login, and auth checks
- [backend/routes_sim.py](backend/routes_sim.py): simulation-frame loading and SSE streaming
- [agent-gateway/](agent-gateway/): Pi-based public agent gateway and OpenAI-compatible API
- [tools/GeneralTool.py](tools/GeneralTool.py): legacy callable workspace, web, academic, and RAG utilities
- [tools/AnalysisTool.py](tools/AnalysisTool.py): legacy callable statistics and chart utilities
- [frontend/src/components/ChatView.vue](frontend/src/components/ChatView.vue): main chat UI
- [frontend/src/components/AboutView.vue](frontend/src/components/AboutView.vue): system information page

## RAG and Experiments

The repository includes standalone RAG and experiment scripts, but they are not required before starting the web platform.

Common entry points:

```bash
rtk .venv/bin/python RAG/script/run_rag_pipeline.py
rtk .venv/bin/python training_free_grpo/collect.py
```

Prepare the required services and data for each script before running it.

## Tests

```bash
rtk .venv/bin/pytest
```

Existing tests mainly cover tools and MCP interaction modules.
