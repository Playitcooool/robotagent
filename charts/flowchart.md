# 流程图（Agent 任务处理流程）

```mermaid
flowchart TD
    start([用户输入任务]) --> auth[后端鉴权与会话加载]
    auth --> parse[主 Agent 理解任务]
    parse --> route{任务类型判断}

    route -->|仿真任务| sim[仿真代理]
    route -->|数据分析| anal[分析代理]
    route -->|信息检索| rag[检索增强]
    route -->|直接回答| direct[直接回复]

    sim --> mcp[调用 MCP 工具]
    anal --> tools[数据分析工具]
    rag --> qdrant[查询向量知识库]

    mcp --> frames[写入最新帧状态]
    frames --> sse[SSE 推送前端]
    sse --> render[前端实时渲染]

    tools --> summarize[结果汇总]
    qdrant --> summarize
    direct --> summarize

    summarize --> resp[主 Agent 生成回复]
    resp --> end([返回用户])

    style sim fill:#e3f2fd,stroke:#1976d2
    style anal fill:#fff3e0,stroke:#f57c00
    style rag fill:#e8f5e9,stroke:#388e3c
```
