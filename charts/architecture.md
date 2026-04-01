# 架构图（Mermaid）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3EDF7', 'primaryTextColor': '#2C3E50', 'primaryBorderColor': '#5D7B9D', 'lineColor': '#5D7B9D', 'fontFamily': 'Arial', 'fontSize': '14px'}}}%%
flowchart TB
  subgraph Client[Client]
    browser[browser]
  end

  subgraph Frontend[Frontend]
    ui[登录/聊天/工具结果/实时画面面板]
  end

  subgraph Backend[Backend]
    api[/auth /chat /sim]
    agent[多代理/工具编排]
    stream[SSE 流式推送]
  end

  subgraph Infra[Infra]
    redis_auth[(Redis DB2 用户/会话)]
    redis_chat[(Redis DB1 聊天记录)]
    qdrant[(Qdrant 向量知识库)]
    sim_dir[(共享帧目录 mcp/.sim_stream)]
  end

  subgraph MainAgent[MainAgent]
    main_llm[Qwen3.5-9B-Claude-4.6-HighIQ<br/>Ollama 本地部署]
  end

  subgraph SubAgents[SubAgents]
    sim_agent[仿真代理]
    anal_agent[分析代理]
  end

  subgraph MCP[MCP]
    pyb[PyBullet 仿真]
    tools[MCP 工具集]
  end

  browser --> ui
  ui --> api
  api --> agent
  agent --> main_llm
  main_llm --> sim_agent
  main_llm --> anal_agent
  sim_agent --> tools
  tools --> pyb
  pyb --> sim_dir
  api --> redis_auth
  api --> redis_chat
  main_llm --> qdrant
  api --> stream
  stream --> ui
  api --> sim_dir
  sim_dir --> stream
  api --> redis_auth

  classDef client fill:#FFF3CD, stroke:#F9A825
  classDef service fill:#E3F2FD, stroke:#1976D2
  classDef subagent fill:#FFF3E0, stroke:#F57C00
  classDef infra fill:#E8F5E9, stroke:#388E3C

  class browser,ui client
  class api,agent,stream,main_llm,Frontend service
  class sim_agent,anal_agent,pyb,tools,MCP subagent
  class redis_auth,redis_chat,qdrant,sim_dir,Infra infra
```
