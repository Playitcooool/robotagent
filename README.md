# RobotAgent 🦾

**状态**: 进行中 — 已实现基础代理接入、流式聊天接口、分析工具骨架与系统 prompts。

---

## 概览 🔍
一个轻量化的本地/自托管多代理项目，目标是通过 Ollama（或其他本地 LLM）驱动的代理执行任务（分析、仿真与总协调）。包含：

- 后端：FastAPI 服务 (`server.py`)，提供流式聊天接口（NDJSON）与基本健康检查点。✅
- 主代理：位于 `main.py`，默认使用 `langchain_ollama.ChatOllama` 创建 agent。✅
- 子代理提示（system prompts）：位于 `sys_prompts/`（`AnalysisAgentPrompt.py`, `MainAgentPrompt.py`, `SimulationAgentPrompt.py`），已补充清晰的中文系统提示。✅
- 工具：`tools/AnalysisTool.py`（数据分析 + 可视化接口，保存到 `output/analysis/`，返回 base64 图像），以及通用工具 `tools/general_tools.py`。✅
- 前端：Vue 3（`frontend/`），已有聊天界面 `App.vue`，支持从后端流式接收增量回复并实时渲染。✅

---

## 快速开始 🚀

### 环境依赖
- Python 3.10+、conda/venv（建议）
- 推荐安装（视需要）: `pandas`, `matplotlib`, `seaborn`, `wordcloud`（用于 `tools/AnalysisTool.py`）
- Ollama 服务：本地或远端 Ollama API（默认 `http://localhost:11434`）

示例（conda）:

```bash
conda create -n robotagent python=3.10 -y
conda activate robotagent
pip install -r requirements.txt  # 若项目添加了 requirements
```

> 注意：`langchain_ollama` 的安装与 Ollama 服务配置请参阅 Ollama 官方文档。


### 启动后端

```bash
# 在项目根目录
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

环境变量（可选）:
- `OLLAMA_BASE_URL` — Ollama 服务地址（默认 `http://localhost:11434`）
- `OLLAMA_MODEL` — 默认模型（例如 `qwen3:8b`）

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开浏览器访问前端地址（Vite 会显示），可以发送消息并观察流式响应。

---

## 流式聊天接口（实现细节） 🔁
后端 `/api/chat/send` 返回 `application/x-ndjson`（newline-delimited JSON），每一行是一个 JSON 对象：

- `{ "type": "delta", "text": "..." }` — 部分或最新的回复文本（前端用以更新占位的 assistant 消息）。
- `{ "type": "error", "error": "..." }` — 出错信息。
- `{ "type": "done" }` — 完成信号。

前端在 `frontend/src/App.vue` 中使用 `res.body.getReader()` 逐行读取并增量更新消息。可以根据需要改成 SSE 或 WebSocket。

---

## 工具与提示（prompts）🧰
- `tools/AnalysisTool.py`：CSV 快速摘要（`summarize_csv`、`describe_stats`）及可视化（直方图、相关矩阵、时间序列、词云），输出文件保存在 `output/analysis/`，并返回 base64 PNG。已用 `langchain_core.tools.tool` 装饰，便于被 agent 调用。
- `sys_prompts/`：为三个子代理（Analysis/Main/Simulation）撰写了结构化中文系统提示，规范输出格式和交互原则（例如 JSON 化输出、证据和置信度）。

---

## 已知限制 & 下一步计划 ✅
- 当前实现依赖本地 Ollama 服务；如果 Ollama 不可用，会在后端返回错误提示，但不会回退到模拟器。可选：添加离线回退策略或托管 LLM 支持。
- 流式协议目前使用 NDJSON；可选择升级为 SSE 或 WebSocket 以支持更复杂的交互（token-by-token、高亮等）。
- 计划：
  - 增加单元测试（`tests/test_analysis_tool.py`）和集成测试覆盖流式路径（待实现）。
  - 添加 `/api/model/status` 健康检查端点并在启动时验证 Ollama 可达性。
  - 改进前端 UX（加载指示、流速控制、错误重试）。

---

## 常见问题与调试提示 🛠️
- 后端无法访问 Ollama：确认 `OLLAMA_BASE_URL`、`OLLAMA_MODEL` 与 Ollama 服务运行情况（`curl` 验证）。
- 可视化不生成：检查是否安装 `matplotlib` 并且 `output/analysis/` 可写。

---

如果希望，我可以：
- 添加 `requirements.txt` / `environment.yml`，并为 `frontend` 添加 `proxy` 到 `/api`（开发时免跨域配置）。
- 添加健康检查端点和更完善的单元测试。

如需我继续，上面哪一项优先？ (例如：添加测试、添加模型健康检查、或改为 SSE 流式)。