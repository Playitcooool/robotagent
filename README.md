# RobotAgent

RobotAgent 是一个带工具调用能力的多代理系统，包含：

- FastAPI 后端（聊天、会话、鉴权、实时仿真帧转发）
- Vue 3 前端（登录注册、聊天、工具结果、实时 PyBullet 画面）
- MCP PyBullet 服务（创建/操作仿真环境并持续输出实时帧）
- SFT 数据与训练脚本（含 training-free GRPO 流程）

## 功能概览

- Deep Agent 对话与工具调用
- 登录/注册与登录态保持（token + Redis session）
- 聊天记录按用户隔离存储
- MCP 工具执行时实时渲染仿真画面（右侧面板）
- SFT 训练与 Training-Free GRPO 轨迹流程

## 目录结构

- `server.py`: 主后端服务（FastAPI）
- `frontend/`: Vue 3 前端
- `mcp/mcp_server.py`: PyBullet MCP 服务
- `config/config.yml`: LLM 与 MCP 配置
- `SFT/train.py`: SFT 训练脚本
- `SFT/sample_trajectory.py`: 轨迹采样脚本
- `SFT/training_free_grpo/`: collect / score / summarize 三段脚本

## 运行前准备

### 1) Python 与 Node

建议：

- Python 3.10+
- Node.js 18+

### 2) Redis

后端默认使用 3 个 Redis DB（同一实例，不同库）：

- `redis://127.0.0.1:6379/0`: 现有默认 Redis（保留）
- `redis://127.0.0.1:6379/1`: 聊天记录
- `redis://127.0.0.1:6379/2`: 用户与登录会话

快速启动（Docker）：

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### 3) Python 依赖

本仓库未提供统一 `requirements.txt`，请按你当前环境安装项目所需依赖（例如 FastAPI、langchain、deepagents、redis、pybullet、numpy 等）。

### 4) 前端依赖

```bash
cd frontend
npm install
```

## 配置说明

### 模型与 MCP 配置

编辑 `config/config.yml`：

- `model_url`: OpenAI-compatible 模型服务地址（如 LM Studio）
- `llm.*`: 各角色模型名
- `mcp.ip` 与 `mcp.port`: MCP 服务地址（默认 `http://localhost:8001`）

### 实时仿真帧共享目录（关键）

后端 `server.py` 与 MCP `mcp/mcp_server.py` 通过同一目录共享帧文件。

请确保两边 `PYBULLET_STREAM_DIR` 一致。

默认：

- 后端：`<repo>/mcp/.sim_stream`
- MCP：`<repo>/mcp/.sim_stream`

如果你自定义目录，需要同时设置两端环境变量。

## 启动顺序（推荐）

在项目根目录执行。

### 1) 启动 MCP 服务（PyBullet）

方式 A：直接运行

```bash
python mcp/mcp_server.py
```

方式 B：Docker Compose

```bash
cd docker
docker-compose up -d --build
```

### 2) 启动后端

```bash
python server.py
```

默认监听：`http://0.0.0.0:8000`

### 3) 启动前端

```bash
cd frontend
npm run dev
```

打开 Vite 输出的地址（通常 `http://localhost:5173`）。

## 登录与会话

前端已内置登录/注册页：

- 首次使用先注册
- 登录成功后 token 存在 `localStorage`
- 刷新页面后会通过 `/api/auth/me` 自动恢复登录态
- 退出登录调用 `/api/auth/logout`

后端鉴权接口：

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

聊天与历史接口需要 `Authorization: Bearer <token>`。

## 实时 PyBullet 画面说明

- MCP 工具在执行过程中持续写入 `latest.png` + `latest.json`
- 后端 `/api/sim/stream` 通过 SSE 推送最新帧
- 前端右侧 `ToolResults` 实时显示 `image_url`

典型流程：

1. 在聊天中触发仿真类工具（如推箱子）
2. 右侧实时显示创建环境、创建物体、动作执行过程
3. 本次动作结束后保留最后一帧
4. 下一次动作继续在同一仿真环境上执行（除非显式清理）

清理仿真环境可调用 MCP 的 `cleanup_simulation_tool`。

## SFT 与 Training-Free GRPO

### SFT 训练

```bash
python SFT/train.py --data trajectories.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --output sft_ckpt
```

### 轨迹采样

```bash
python SFT/sample_trajectory.py
```

### Training-Free GRPO（在线经验闭环）

```bash
python training_free_grpo/collect.py
```

`collect.py` 已内置每轮闭环：采样 -> 打分 -> 总结单条经验 -> 写入经验库 -> 经验注入下一轮 system prompt。

详细参数见：`training_free_grpo/README.md`。

## 常见问题

### 1) 前端看不到实时仿真画面

优先检查：

- MCP 是否已启动
- `PYBULLET_STREAM_DIR` 两端是否一致
- `mcp/.sim_stream/latest.json` 与 `latest.png` 是否在更新
- 后端 `/api/sim/debug` 返回是否正常

### 2) 登录后请求 401

- token 可能过期，重新登录
- 检查浏览器本地是否有 `robotagent_auth_token`
- 检查后端 `AUTH_REDIS_URL` 连接状态

### 3) 模型不返回内容

- 检查 `config/config.yml` 的 `model_url` 与模型名
- 确认本地模型服务已启动并支持 OpenAI-compatible API

## 开发提示

- 前端构建：`cd frontend && npm run build`
- 后端语法检查：`python -m py_compile server.py`
- MCP 服务默认端口：`8001`
- 后端默认端口：`8000`
