# 数据流图（Mermaid, Level 0）

```mermaid
flowchart LR
  user[用户]:::ext
  frontend[前端 Vue3]:::proc
  backend[后端 FastAPI]:::proc
  mcp[MCP PyBullet 服务]:::proc
  model[模型服务
(OpenAI-compatible)]:::ext

  redis_chat[(Redis DB1
聊天记录)]:::store
  redis_auth[(Redis DB2
用户/会话)]:::store
  sim_dir[(共享帧目录
mcp/.sim_stream)]:::store

  user -->|登录/聊天/查看| frontend
  frontend -->|API 请求| backend
  backend -->|鉴权/会话| redis_auth
  backend -->|聊天记录| redis_chat
  backend -->|模型调用| model

  backend -->|工具调用| mcp
  mcp -->|latest.png/json| sim_dir
  backend -->|SSE/拉取| sim_dir
  backend -->|实时帧| frontend
  frontend -->|画面展示| user

  classDef ext fill:#fff3cd,stroke:#555,stroke-width:1px;
  classDef proc fill:#e8f0fe,stroke:#1a73e8,stroke-width:1px;
  classDef store fill:#e6f4ea,stroke:#137333,stroke-width:1px;
```
