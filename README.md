# RobotAgent

基于多智能体协同的机器人仿真任务规划与执行系统。

## 系统概述

RobotAgent 是一个多代理聊天系统，支持工具调用、会话管理、认证登录和仿真画面实时流。系统基于 LangChain 和 ReAct 架构构建，由主代理、仿真代理和分析代理组成协作闭环，通过 MCP（Model Context Protocol）接入 PyBullet/Gazebo 仿真工具，通过 Agentic RAG 增强领域知识理解，并通过 Vue3 + FastAPI 提供端到端可视化平台。

**核心特性：**
- 多代理协同：主代理（任务拆解与路由）+ 仿真代理（执行仿真动作）+ 分析代理（数据统计与可视化）
- 全本地部署：所有代理使用 Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16，通过 OMLX + LM Studio 本地运行
- Agentic RAG：检索增强作为可按需调用的工具，提升复杂指令理解质量
- Training-Free GRPO：通过轨迹采样与经验回灌实现无参数更新的策略优化
- 实时可视化：仿真帧 SSE 推送，前端实时渲染执行过程

## 目录结构

```
robotagent/
├── server.py              # 后端统一入口
├── dev.sh                # 一键启动前后端
├── config/config.yml     # 模型与 MCP 基础配置
│
├── backend/             # 后端（FastAPI）
│   ├── app.py           # 主应用装配 + 聊天/会话接口
│   ├── routes_auth.py   # 认证与登录接口
│   ├── routes_sim.py     # 仿真帧接口（debug/latest/stream）
│   ├── schemas.py       # Pydantic 请求模型
│   ├── auth_utils.py    # 认证辅助函数
│   └── stream_utils.py  # 流式消息解析与文本提取
│
├── frontend/            # 前端（Vue 3 + Vite）
│   └── src/            # Vue 组件、视图、状态管理
│
├── mcp/                # MCP 服务
│   ├── mcp_server.py   # PyBullet 仿真服务
│   └── gazebo_mcp_server.py  # Gazebo 仿真服务
│
├── tools/              # 工具集
│   ├── GeneralTool.py  # 通用工具（搜索/文件/格式化）
│   ├── AnalysisTool.py # 分析工具（统计/绘图）
│   └── SubAgentTool.py# 子代理工具
│
├── prompts/            # Agent System Prompt
│   ├── MainAgentPrompt.py
│   ├── SimulationAgentPrompt.py
│   └── AnalysisAgentPrompt.py
│
├── RAG/               # 检索增强生成模块
│   ├── query.py       # 在线检索入口
│   ├── script/        # 建库脚本
│   │   ├── download_arxiv_pdfs.py
│   │   ├── run_rag_pipeline.py
│   │   └── write_qdrant.py
│   └── eval/         # RAG 评估数据
│
├── training_free_grpo/ # Training-Free GRPO 经验优化
│   └── collect.py     # 在线轨迹采集与经验回灌
│
├── experiments/       # 实验评估
│   ├── evaluate_experiment_01.py  # 学术搜索问答质量评估
│   ├── evaluate_experiment_02.py  # 仿真任务执行质量评估
│   ├── evaluate_experiment_03.py  # 尝试次数与成功率分析
│   └── config.yaml
│
├── docker/            # Docker 容器化配置
│
├── charts/            # Mermaid 流程图源码
│
└── trajectories.jsonl # 仿真任务轨迹数据
```

## 环境要求

- Python 3.10+
- Node.js 18+
- Redis 6+
- OMLX（MLX 本地推理，支持 OpenAI-compatible API）
- Mac M4（24GB 统一内存，推荐）

## 依赖安装

### 前端

```bash
cd frontend && npm install
```

### 后端

项目未提供统一 `requirements.txt`，请按当前环境安装核心依赖：

```bash
pip install fastapi uvicorn langchain langchain-openai redis deepagents
```

## 配置

编辑 `config/config.yml`：

```yaml
llm: "Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16"
model_url: "http://localhost:1234/v1"
api_key: "no_need"

analysis_llm: "Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16"
analysis_model_url: "http://localhost:1234/v1"
analysis_api_key: "no_need"

simulation_llm: "Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16"
simulation_model_url: "http://localhost:1234/v1"
simulation_api_key: "no_need"

mcp:
  ip: "http://localhost"
  port: "18001"

tavily:
  api_key: "tvly-dev-xxx"
```

后端默认 Redis：
- `redis://127.0.0.1:6379/1`：聊天记录
- `redis://127.0.0.1:6379/2`：认证与会话

## 启动

### 1. 启动 OMLX 本地推理服务

使用 OMLX 加载 `Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16`，启动为 OpenAI-compatible API 服务（默认 localhost:1234）。

### 2. 启动 MCP 仿真服务

```bash
# PyBullet
python mcp/mcp_server.py

# 或 Gazebo
python mcp/gazebo_mcp_server.py
```

Docker 方式（推荐）：

```bash
# PyBullet MCP（映射到宿主机 8001）
docker compose -f docker/pybullet/docker-compose.yml up -d --build

# Gazebo MCP（映射到宿主机 8002）
docker compose -f docker/gazebo/docker-compose.yml up -d --build
```

### 3. 启动前后端（推荐）

```bash
./dev.sh
```

会同时启动：
- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

指定后端 Python 环境：

```bash
BACKEND_PYTHON=/path/to/venv/bin/python ./dev.sh
```

## 主要接口

### 认证

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

### 聊天与会话

- `GET /api/messages` — 按会话拉取历史
- `GET /api/sessions` — 会话列表
- `DELETE /api/sessions/{session_id}` — 删除会话
- `POST /api/chat/send` — 聊天发送（NDJSON 流式输出）

### 仿真

- `GET /api/sim/debug` — 仿真状态调试
- `GET /api/sim/latest-frame` — 最新帧元数据
- `GET /api/sim/latest.png` — 最新帧图像
- `GET /api/sim/stream` — SSE 帧推送

## MCP 工具

### PyBullet 仿真工具（19个）

| 工具名称 | 功能描述 |
|---|---|
| `initialize_simulation` | 初始化 PyBullet 仿真环境 |
| `check_static_assets` | 检查静态资源是否完整 |
| `push_cube_step` | 推送立方体（指定起点、推动向量、步数） |
| `grab_and_place_step` | 抓取并放置物体 |
| `path_planning` | 路径规划（线性插值） |
| `adjust_physics` | 调整物理参数（摩擦系数、弹性系数） |
| `multi_object_grab_and_place` | 多物体抓取放置 |
| `simulate_vision_sensor` | 模拟视觉传感器拍照 |
| `cleanup_simulation_tool` | 清理仿真环境 |
| `check_simulation_state` | 检查当前仿真状态 |
| `reset_simulation` | 重置仿真世界 |
| `pause_simulation` | 暂停仿真 |
| `unpause_simulation` | 恢复仿真 |
| `get_object_state` | 获取物体状态（位置、速度） |
| `set_object_position` | 设置物体位置 |
| `step_simulation` | 执行指定步数 |
| `create_object` | 创建物体（cube/sphere/cylinder） |
| `delete_object` | 删除物体 |
| `set_gravity` | 设置重力 |

### Gazebo 仿真工具（17个）

| 工具名称 | 功能描述 |
|---|---|
| `initialize_ros_connection` | 初始化 ROS2 连接 |
| `spawn_model` | 创建模型（支持 URDF/SDF） |
| `list_builtin_models` | 列出内置模型 |
| `delete_model` | 删除模型 |
| `get_model_state` | 获取模型状态 |
| `set_model_state` | 设置模型状态 |
| `list_models` | 列出所有模型 |
| `pause_simulation` | 暂停仿真 |
| `unpause_simulation` | 恢复仿真 |
| `reset_simulation` | 重置仿真（时间和状态） |
| `reset_world` | 重置世界（仅状态） |
| `capture_camera` | 捕获相机图像 |
| `cleanup_ros_connection` | 清理 ROS 连接 |
| `get_simulation_info` | 获取仿真信息 |
| `apply_force` | 施加外力 |
| `move_object` | 移动物体（简化版） |
| `create_simple_object` | 创建简单几何体 |

## RAG 建库

一键 RAG Pipeline（推荐）：

```bash
python RAG/script/run_rag_pipeline.py
```

默认流程：arXiv PDF 下载 → MinerU 解析 → Markdown 切分 → chunks 写入 Qdrant。

可选写入 Qdrant：

```bash
python RAG/script/run_rag_pipeline.py --to-qdrant
```

## Training-Free GRPO

在线轨迹采集与经验回灌：

```bash
python training_free_grpo/collect.py \
  --base_url http://localhost:1234/v1 \
  --model "Jackrong:MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-bf16" \
  --api_key no_need \
  --prompts SFT/data.txt \
  --output_path output/training_free_grpo/trajectories.jsonl \
  --score_path output/training_free_grpo/trajectory_scores.jsonl \
  --memory_json_path output/training_free_grpo/external_memory.json \
  --memory_md_path output/training_free_grpo/external_memory.md \
  --samples_per_prompt 3 \
  --max_experiences_in_prompt 20
```

每轮执行：采样轨迹 → 评分 → 经验总结 → 写入 external_memory → 注入下一轮 system prompt。

## 实验评估

```bash
# 实验1: Agentic 学术搜索问答质量评估
python experiments/evaluate_experiment_01.py \
    --queries experiments/data/rag_queries.jsonl \
    --out-dir results/exp01

# 实验2: 机器人仿真任务执行质量评估
python experiments/evaluate_experiment_02.py \
    --trajectories trajectories.jsonl \
    --out-dir results/exp02

# 实验3: 尝试次数与成功率关系分析
python experiments/evaluate_experiment_03.py \
    --trajectories trajectories.jsonl \
    --out-dir results/exp03
```

每个实验输出：`details.jsonl`（详细结果）、`summary.json`（汇总统计）、`figures/`（可视化图表）。

外部 Judge 使用 DeepSeek，需在 `experiments/config.yaml` 配置：

```yaml
judge:
  api_base: "https://api.deepseek.com"
  api_key: "sk-xxx"
  model: "deepseek-chat"
  timeout: 120
```

## 常见问题

### 1) 重新运行 `./dev.sh` 报端口占用

`dev.sh` 已内置端口清理，直接再次运行即可自动重启。

### 2) 前端能聊天但无仿真画面

检查：
- MCP 服务是否已启动
- 后端能否访问 MCP 端口（默认 localhost:18001）
- `/api/sim/debug` 是否显示帧文件存在

### 3) 模型调用超时

检查 OMLX 服务是否正常运行，localhost:1234 是否可访问。可通过 `curl http://localhost:1234/v1/models` 验证。

## 参考资料

- LangChain: https://python.langchain.com/
- MCP: https://modelcontextprotocol.org/
- PyBullet: https://pybullet.org/
- OMLX: 本地 MLX 推理服务（配置于 localhost:1234/v1）
- OMLX/MLX: https://github.com/ml-explore/mlx
