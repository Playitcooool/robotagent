# Config Guide

This folder owns runtime configuration.

- `config.yml` defines model endpoints, API keys, MCP service addresses, Tavily settings, and judge/evaluation configuration.

Avoid committing real secrets. Code should read model names, base URLs, MCP URLs, and external service settings from this file or environment variables rather than duplicating them.

When changing config keys, update all consumers in `backend/model_config.py`, `backend/app.py`, `tools/`, `prompts/`, and experiment scripts that read the same keys.
