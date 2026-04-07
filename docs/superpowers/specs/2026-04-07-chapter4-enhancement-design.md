# 第四章内容补全设计文档

## 概述

本次设计针对论文第四章（关键实现细节）的两处内容进行补全和重构：

1. **4.5节** 重构：从"前端交互实现"扩展为含前端渲染与后端架构的完整实现细节
2. **4.6节** 补全：完整阐述 Training-Free GRPO 经验强化机制的 pipeline

---

## 4.5 前端交互实现

### 4.5.1 界面模块

**现状**：仅有概述性描述，缺乏具体组件说明。

**补全内容**：
- 认证视图（登录/注册）
- 会话侧栏（会话列表、创建/切换/删除）
- 对话主视图（消息渲染、流式文本展示）
- 结果面板（仿真帧、视频预览、轨迹回放）
- 工具调用面板（子代理状态）

### 4.5.2 状态管理

**现状**：仅一句话描述"中心状态 + 组件渲染"模式。

**补全内容**：
- Pinia store：会话状态、消息流、仿真帧、工具调用状态
- 多事件流并发管理：文本流（SSE/NDJSON）与仿真流（SSE）的并发启动与协调
- 前端帧缓存与去重：基于 timestamp 的帧跳过逻辑

### 4.5.3 前端仿真渲染技术（新增）

**核心内容**：
- **Canvas 2D 直接渲染**：前端周期性轮询 `/api/sim/frames` 获取最新 PNG 帧，通过 `Image` 对象绘制到 `<canvas>`
- **响应式画布**：根据容器宽度自适应缩放，保持 320×240 原始分辨率
- **帧率控制**：避免丢帧时重绘过频，使用 `requestAnimationFrame` 配合 `lastRenderTime` 节流
- **断线重连**：SSE 流断开时使用指数退避重连（jitter），避免惊群效应

### 4.5.4 后端架构与异步请求（新增）

**核心内容**：

**4.5.4.1 分层架构**
- 接入层（Vue3）、服务层（FastAPI）、智能层（LangGraph Agent）、资源层（Redis / Qdrant / MCP）
- 各层职责边界清晰，层间通过标准化接口通信

**4.5.4.2 异步请求处理**
- FastAPI `async def` 全链路异步：对话、化学术问答、仿真流
- Redis 异步客户端（`redis.asyncio`）存储会话与消息历史
- 长连接超时控制与请求取消（`request.is_disconnected()`）

**4.5.4.3 流式响应机制**
- **LLM 文本流**：通过 `active_agent.astream()` 生成 token 事件，前端通过 `fetch` + `ReadableStream` 消费 NDJSON 格式的增量文本
- **仿真帧流**：SSE（Server-Sent Events）端点 `/api/sim/stream`，后端以自适应轮询频率（50ms/200ms/500ms）推送帧元数据，前端据此轮询帧图片
- **双流并发**：对话消息流与仿真帧流并发启动，保证"语言结果"与"环境反馈"同步到达

**4.5.4.4 仿真帧服务**
- MCP 仿真服务（PyBullet / Gazebo）将帧写入共享目录（`mcp/.sim_stream/latest.png`）
- FastAPI 通过文件系统直接服务 PNG 文件，前端通过 `GET /api/sim/frames` 轮询获取
- 文件写入使用原子操作（写临时文件 + `os.replace`）防止读取到部分写入帧

---

## 4.6 Training-Free GRPO 经验强化机制

### 4.6.1 核心思想

Training-Free GRPO 通过**轨迹比较**而非**模型参数更新**来优化策略。系统每次任务执行生成多条候选轨迹，利用外部 Judge 模型对同组轨迹进行相对打分，提炼出优势策略与失败经验，并在不修改主模型参数的情况下将其回灌至 Agent 上下文。

### 4.6.2 经验轨迹收集

对同一查询 q，系统先采样 G 条轨迹：
```
τᵢ ~ πθ(·|q, Et),  i = 1, …, G
```
- `πθ` 为冻结策略模型
- `Et` 为第 t 轮经验库
- `samples_per_prompt` 默认为 4
- 仿真环境在每次 attempt 后自动重置（`CLEANUP_SIMULATION_PER_ATTEMPT = True`）

每条轨迹提取三类信息：
1. **工具调用序列**：MCP 工具名称与参数
2. **关键中间状态**：仿真环境状态、观测结果
3. **最终结果**：任务完成标志、输出文本

### 4.6.3 轨迹评分

由 Judge LLM 对每条轨迹独立评分，评分维度包括（0-10 分制）：

| 维度 | 说明 |
|------|------|
| task_completion | 任务完成质量 |
| correctness | 结果准确性 |
| clarity | 输出清晰度 |
| robustness | 异常处理质量 |
| conciseness | 简洁性与完整性平衡 |
| overall_score | 综合得分 |

### 4.6.4 经验对比与总结

对高优势（`S_top`）与低优势（`S_bottom`）轨迹进行归因对比，Judge Agent 生成：

- **summary**：1-2 句核心经验总结
- **principles**：3-5 条核心原则
- **dOs**：应该做的事项（最多 5 条）
- **DON'Ts**：应该避免的事项（最多 5 条）

经验更新策略：
```
Et+1 = U(Et, Stop, Sbottom)
```
- 若新旧原则重叠 ≥ 3 条：**跳过**（避免冗余）
- 若重叠 1-2 条：**部分更新**
- 若全新原则：**写入新经验**
- 每类 Agent 最多保留 5 条经验

### 4.6.5 经验存储与注入

**存储结构**（`external_memory.json`）：
```json
{
  "meta": {"method": "training_free_grpo_judge_agent", "updated_at": 123456},
  "experiences": [{
    "id": "1d6ed69f",
    "prompt_id": 1,
    "prompt": "用户原始查询",
    "summary": "核心经验教训",
    "principles": ["原则1", "原则2", "原则3"],
    "dos": ["应做事项1", "应做事项2"],
    "donts": ["应避免事项1"],
    "agent_type": ["simulator", "main"],
    "score": 9.0,
    "created_at": 123456
  }]
}
```

**注入方式**：经验以结构化文本形式追加至系统提示：
```
P(t+1) = P_base ⊕ Format(E(t+1))
```

三类 Agent（main / simulator / data-analyzer）各自维护独立的经验上下文，互不影响。

### 4.6.6 离线转换 Pipeline

1. `collect.py`：在线收集 → 评分 → 总结 → `external_memory.json`
2. `convert_experiences.py`：过滤低分经验（阈值 6.0）→ 按 Agent 类型分组 → `agent_experiences.json`
3. `backend/app.py`：启动时加载 → `_load_agent_experiences()` → 注入各 Agent

---

## 实现计划

### 涉及的文件修改

1. **2236127阮炜慈初稿_更新版.docx**：第 4.5.1、4.5.2、4.6 节内容替换与扩充
2. 替换后新增 4.5.3（前端渲染）、4.5.4（后端架构）两个子节
3. 4.6 节补全完整 GRPO pipeline 描述

### 写入顺序

1. 先补全 4.5.3 和 4.5.4（新增内容）
2. 扩充 4.5.1 和 4.5.2（替换现有内容）
3. 最后补全 4.6 节（替换"具体流程"后的空缺）
