# 时序图（Mermaid）

%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3EDF7', 'primaryTextColor': '#2C3E50', 'primaryBorderColor': '#5D7B9D', 'lineColor': '#5D7B9D', 'fontFamily': 'Arial', 'fontSize': '14px'}}}%%

```mermaid
sequenceDiagram
  participant U as 🟡 用户
  participant FE as 🔵 前端 Vue3
  participant BE as 🔵 后端 FastAPI
  participant Main as 🔵 主代理 Qwen3.5-9B
  participant Sim as 🟠 仿真代理
  participant Anal as 🟠 分析代理
  participant MCP as 🟠 MCP 服务
  participant SIM as 🟢 共享帧目录
  participant RAG as 🟢 向量知识库 Qdrant

  U->>FE: 登录/进入聊天
  FE->>BE: POST /api/auth/login application/json
  alt 鉴权成功
    BE-->>FE: 200 {token: Bearer ...}
  else 鉴权失败
    BE-->>FE: 401 Unauthorized
  end

  U->>FE: 发送聊天
  FE->>BE: POST /api/chat Bearer {token} [timeout=60s]
  BE->>Main: 任务规划与路由 [timeout=30s]
  Main->>RAG: Agentic RAG 检索(如需) [timeout=10s]
  RAG-->>Main: 知识增强结果

  alt 需要仿真任务
    Main->>Sim: 路由到仿真代理
    Sim->>MCP: 调用仿真工具 [timeout=120s]
    MCP-->>Sim: 执行结果
    Sim-->>Main: 执行结果
  else 需要数据分析
    Main->>Anal: 路由到分析代理
    Anal-->>Main: 分析结果
  end

  Main-->>BE: 汇总结果
  BE-->>FE: SSE 推送最新帧 [stream]
  FE-->>U: 实时渲染画面+回复

  Note over BE,Main: 错误处理
  Main-->>BE: ⚠️ 模型超时 [timeout]
  BE-->>FE: 500 任务执行失败
  MCP-->>Sim: ❌ 仿真失败
  Sim-->>Main: Fallback 策略
```
