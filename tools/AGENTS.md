# Tools Guide

This folder contains LangChain tool implementations and sub-agent wiring.

- `GeneralTool.py` exposes general tools such as current time, workspace search, web/academic search, and local RAG retrieval.
- `AnalysisTool.py` provides data analysis helpers for CSV summaries, statistics, and chart generation.
- `SubAgentTool.py` initializes and calls specialized analysis and simulation agents.
- `mcp_loader.py` progressively loads MCP tools so the main agent can use simulation services without hard-coding each tool.

Tool outputs are user-visible through the backend stream and frontend `ToolResults` components, so return structured, compact payloads where possible. Configurable services such as Qdrant, Tavily, model endpoints, and MCP URLs should come from environment variables or `config/config.yml`, not inline constants unless they are safe defaults.

When adding or renaming tools, update the relevant prompt context in `prompts/` and check `/api/tools` behavior in `backend/app.py`.
