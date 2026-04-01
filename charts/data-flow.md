# 数据流图（Mermaid, Level 0）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3EDF7', 'primaryTextColor': '#2C3E50', 'primaryBorderColor': '#5D7B9D', 'lineColor': '#5D7B9D', 'fontFamily': 'Arial', 'fontSize': '14px'}}}%%
flowchart LR
  user[user]:::ext
  frontend[frontend]:::proc
  backend[backend]:::proc
  mcp[mcp]:::proc

  subgraph 模型服务
    main_agent[main_agent]:::proc
    sub_agent[sub_agent]:::proc
  end

  redis_chat[(Redis DB1 聊天记录<br/>{session_id, role, content})]:::store
  redis_auth[(Redis DB2 用户/会话<br/>{token, user_id, expires_at})]:::store
  qdrant[(qdrant)]:::store
  sim_dir[(sim_dir<br/>mcp/.sim_stream<br/>latest.png/json)]:::store

  user -->|登录/聊天/查看| frontend
  frontend -->|API 请求| backend
  backend -->|鉴权/会话| redis_auth
  backend -->|聊天记录| redis_chat
  backend -->|任务规划| main_agent
  main_agent -->|工具调用| sub_agent
  sub_agent -->|检索增强| qdrant
  backend -->|工具调用| mcp
  mcp -->|latest.png/json| sim_dir
  backend -->|SSE/拉取| sim_dir
  backend -->|实时帧| frontend
  frontend -->|画面展示| user

  backend -.->|401 鉴权失败| frontend
  redis_auth -.->|连接失败| backend
  redis_chat -.->|连接失败| backend
  qdrant -.->|不可用| main_agent
  mcp -.->|超时/失败| backend
  backend -.->|SSE中断 降级轮询| frontend

  classDef ext fill:#FFF3CD, stroke:#F9A825, stroke-width:2px;
  classDef proc fill:#E3F2FD, stroke:#1976D2, stroke-width:2px;
  classDef store fill:#E8F5E9, stroke:#388E3C, stroke-width:2px;
  classDef err fill:#FFEBEE, stroke:#D32F2F, stroke-dasharray:5, stroke-width:2px;
```
