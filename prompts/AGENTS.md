# Prompts Guide

This folder controls the behavior and context of the agents.

- `MainAgentPrompt.py` builds the primary system prompt for task understanding, tool use, planning, and sub-agent coordination.
- `AnalysisAgentPrompt.py` guides the data-analysis sub-agent.
- `SimulationAgentPrompt.py` guides the simulation sub-agent.
- `context_loader.py` loads additional project or robot context for prompt construction.

Prompt changes can alter tool selection, streaming behavior, and downstream simulation commands. Keep instructions concrete and compatible with the actual tools in `tools/` and MCP capabilities in `mcp/`.

Do not bake API keys, model names, or endpoint assumptions into prompts. Prefer references to configured services and available tools.
