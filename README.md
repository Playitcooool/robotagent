# RobotAgent

RobotAgent 是一个多代理聊天系统，支持工具调用、会话管理、认证登录和仿真画面实时流。

## 当前架构

- 后端：FastAPI（`backend/` 模块化）
- 前端：Vue 3 + Vite（`frontend/`）
- 仿真服务：MCP（PyBullet / Gazebo）
- 存储：Redis（会话/聊天/认证）
- 模型接入：OpenAI-compatible API（例如 LM Studio）

## 目录结构（最新）

- `server.py`：后端统一入口（转发到 `backend.app:app`）
- `dev.sh`：一键启动前后端（支持重复执行自动重启）
- `backend/app.py`：主应用装配 + chat/messages/sessions 相关接口
- `backend/routes_auth.py`：认证与登录相关接口
- `backend/routes_sim.py`：仿真帧接口（debug/latest/stream）
- `backend/schemas.py`：Pydantic 请求模型
- `backend/auth_utils.py`：认证辅助函数
- `backend/stream_utils.py`：流式消息解析与文本提取
- `frontend/`：前端工程
- `mcp/`：MCP 服务代码
- `config/config.yml`：模型与 MCP 基础配置

## 环境要求

- Python 3.10+
- Node.js 18+
- Redis 6+

## 依赖安装

### 前端

```bash
cd frontend
npm install
```

### 后端

项目未提供统一 `requirements.txt`，请按你当前环境安装依赖（FastAPI、uvicorn、langchain、redis、deepagents 等）。

## 配置

编辑 `config/config.yml`：

- `model_url`：OpenAI-compatible 地址（例如 `http://localhost:1234/v1`）
- `llm.chat / llm.analysis / llm.simulation`：模型名

后端默认 Redis：

- `redis://127.0.0.1:6379/1`：聊天记录
- `redis://127.0.0.1:6379/2`：认证与会话

## 启动（推荐）

在项目根目录：

```bash
./dev.sh
```

会同时启动：

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

### 指定后端 Python（例如 conda 环境）

```bash
BACKEND_PYTHON=/opt/miniconda3/envs/langchain/bin/python ./dev.sh
```

### 后端单独启动

```bash
python server.py
```

## MCP 服务

根据你的使用场景单独启动对应 MCP：

- `mcp/mcp_server.py`（PyBullet）
- `mcp/gazebo_mcp_server.py`（Gazebo）

### Docker 启动（推荐）

#### 启动 PyBullet MCP（映射到宿主机 `8001`）

```bash
docker compose -f docker/pybullet/docker-compose.yml up -d --build
```

- 容器内监听：`8000`
- 宿主机访问：`http://127.0.0.1:8001/mcp`

#### 启动 Gazebo MCP（映射到宿主机 `8002`）

```bash
docker compose -f docker/gazebo/docker-compose.yml up -d --build
```

- 容器内监听：`8002`
- 宿主机访问：`http://127.0.0.1:8002/mcp`

#### 查看状态与日志

```bash
docker ps
docker logs -f docker-robotagent-1
docker logs -f gazebo-gazebo-mcp-1
```

如果你已修改 compose 中的 `container_name` 或 project 名称，请按实际容器名查看日志。

## 主要后端接口

### 通用

- `GET /api/ping`

### 认证（`backend/routes_auth.py`）

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

### 聊天与会话（`backend/app.py`）

- `GET /api/messages`
- `GET /api/sessions`
- `DELETE /api/sessions/{session_id}`
- `POST /api/chat/send`（NDJSON 流式输出）

### 仿真（`backend/routes_sim.py`）

- `GET /api/sim/debug`
- `GET /api/sim/latest-frame`
- `GET /api/sim/latest.png`
- `GET /api/sim/stream`（SSE）

## 前端说明

前端包含：

- 登录/注册
- 左侧会话列表（支持删除会话）
- 中间聊天区（流式回答）
- 右侧工具结果与仿真画面

## 开发与调试

### 前端构建检查

```bash
cd frontend && npm run build
```

### 后端语法检查

```bash
python -m py_compile server.py backend/app.py backend/routes_auth.py backend/routes_sim.py
```

### 流字段调试（可选）

```bash
DEBUG_STREAM_FIELDS=1 BACKEND_PYTHON=/opt/miniconda3/envs/langchain/bin/python ./dev.sh
```

用于查看前几个流式 chunk 的字段情况，便于确认是否有 reasoning/thinking 字段透传。

## RAG 论文抓取方法（按年份均衡 + 质量过滤）

`RAG/script/download_arxiv_pdfs.py` 当前采用的是“边筛边下”的抓取方式，并针对机器人领域做了质量优化：

- 近 N 年均衡采样（默认近 5 年），按年分配目标数量，避免只抓到最新论文
- 每年先过筛再打分，按质量分 Top-K 下载
- 已下载 PDF 自动跳过（文件存在且大于 100KB）
- 实时输出进度与按年份统计，便于实验复现与论文写作

### 一键 Pipeline（推荐）

统一入口脚本：`RAG/script/run_rag_pipeline.py`

```bash
python RAG/script/run_rag_pipeline.py
```

默认流程：

- 下载 arXiv PDF（按年份均衡 + 质量过滤）
- PDF 分批
- MinerU 解析
- 提取 Markdown
- 切分 chunks

可选写入 Qdrant：

```bash
python RAG/script/run_rag_pipeline.py --to-qdrant
```

质量分主要参考：

- 关键词命中（robot/manipulation/motion planning/sim2real 等）
- 正向信号词（benchmark/dataset/ablation/evaluation/open-source 等）
- 负向信号词（position paper/extended abstract 等）
- 分类匹配度（如 `cs.RO`）
- 摘要长度与信息密度

### 推荐命令

```bash
python RAG/script/download_arxiv_pdfs.py
```

如果你在脚本里调整参数，常用项如下：

- `max_results`：总目标下载量
- `years_back`：回溯年份范围（例如 5 或 8）
- `per_year_overfetch`：每年候选过采样倍数（越大越容易筛到高质量）
- `min_quality_score`：质量阈值（越高越严格）

### 输出统计口径

脚本会输出总量与分年份统计：

- 总体：`Selected / ok / failed`
- 按年份：`target / scanned / qualified / selected / ok / failed`

其中 `qualified` 表示通过质量阈值的候选数量。

## MCP 工具列表

### PyBullet 仿真工具 (19个)

| 工具名称 | 功能描述 |
|----------|----------|
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

### Gazebo 仿真工具 (17个)

| 工具名称 | 功能描述 |
|----------|----------|
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

## 常见问题

### 1) 重新运行 `./dev.sh` 报端口占用

`dev.sh` 已内置端口清理。直接再次运行即可自动重启。

### 2) 前端能聊天但无仿真画面

检查：

- MCP 服务是否已启动
- 后端能否访问 MCP 端口
- `/api/sim/debug` 是否显示帧文件存在

### 3) 看不到独立 thinking 字段

若 `astream` 的 chunk 只有 `content`，没有 `content_blocks/additional_kwargs.reasoning*`，说明当前模型/网关没有透传 reasoning 通道。
