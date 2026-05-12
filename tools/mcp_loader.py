"""Progressive MCP tool loader.

Splits MCP tools into core (always loaded with full schema) and extended
(accessible via meta-tools to reduce context window pressure).
"""

import asyncio
import json
import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
import yaml

logger = logging.getLogger("uvicorn")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT_DIR, "config", "config.yml"), "r", encoding="utf-8") as f:
    _config = yaml.load(f.read(), yaml.FullLoader)

# Core tools: always loaded with full schema
# All simulation tools are composition-friendly primitives. The previous "demo-style"
# high-level tools (grab_and_place_step, push_cube_step, path_planning, etc.) were
# removed entirely from the MCP server as they confused the model with internal
# loadURDF calls incompatible with composition.
CORE_TOOL_NAMES = {
    "initialize_simulation",
    "create_object",
    "load_urdf",
    "set_object_position",
    "step_simulation",
    "get_object_state",
    "delete_object",
    "get_simulation_info",
    "check_simulation_state",
    "cleanup_simulation_tool",
}

_mcp_tools_cache: list[BaseTool] | None = None
_mcp_lock = asyncio.Lock()


def _resolve_mcp_urls() -> dict[str, str]:
    mcp_cfg = _config.get("mcp") or {}
    servers = mcp_cfg.get("servers")
    if isinstance(servers, dict) and servers:
        resolved = {}
        for name, url in servers.items():
            if not url:
                continue
            url = str(url).strip()
            if not url.endswith("/mcp"):
                url = url.rstrip("/") + "/mcp"
            resolved[name] = url
        if resolved:
            return resolved
    base = str(mcp_cfg.get("ip") or "http://127.0.0.1").rstrip("/")
    port = str(mcp_cfg.get("port") or "18001")
    urls = {"pybullet": f"{base}:{port}/mcp"}
    gazebo_port = mcp_cfg.get("gazebo_port")
    if gazebo_port:
        urls["gazebo"] = f"{base}:{gazebo_port}/mcp"
    return urls


async def _load_all_mcp_tools() -> list[BaseTool]:
    """Load all MCP tools from configured services (cached)."""
    global _mcp_tools_cache
    async with _mcp_lock:
        if _mcp_tools_cache is not None:
            return _mcp_tools_cache

        service_urls = _resolve_mcp_urls()
        logger.info(f"MCP service endpoints: {service_urls}")
        all_tools: list[BaseTool] = []
        max_retries = int(os.environ.get("SIM_MCP_MAX_RETRIES", "8"))
        retry_delay = float(os.environ.get("SIM_MCP_RETRY_DELAY_SECONDS", "1.0"))

        for name, url in service_urls.items():
            tools = None
            for attempt in range(1, max_retries + 1):
                try:
                    client = MultiServerMCPClient({name: {"transport": "http", "url": url}})
                    tools = await client.get_tools(server_name=name)
                    break
                except Exception as e:
                    logger.warning(f"{name} MCP attempt {attempt}/{max_retries}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
            if tools:
                for t in tools:
                    t.handle_tool_error = True
                all_tools.extend(tools)
            else:
                logger.error(f"{name} MCP unavailable after {max_retries} attempts")

        _mcp_tools_cache = all_tools
        logger.info(f"MCP tools loaded: total={len(all_tools)}")
        return all_tools


async def load_mcp_tools_progressive() -> list[BaseTool]:
    """Load MCP tools with progressive strategy.

    Returns a list of tools for the main agent:
    - Core tools: full schema, directly callable
    - Two meta-tools: list_available_tools + call_extended_tool
    """
    all_tools = await _load_all_mcp_tools()
    if not all_tools:
        return []

    core_tools: list[BaseTool] = []
    extended_tools: list[BaseTool] = []

    for tool in all_tools:
        if tool.name in CORE_TOOL_NAMES:
            core_tools.append(tool)
        else:
            extended_tools.append(tool)

    # Build extended tool registry
    extended_registry: dict[str, BaseTool] = {t.name: t for t in extended_tools}

    def _list_available_tools() -> str:
        """List all extended simulation tools available for on-demand use."""
        lines = []
        for t in extended_tools:
            desc = (t.description or "")[:80]
            lines.append(f"- {t.name}: {desc}")
        return "\n".join(lines) if lines else "No extended tools available."

    async def _call_extended_tool(name: str, args_json: str = "{}") -> str:
        """Call an extended simulation tool by name with JSON arguments."""
        tool = extended_registry.get(name)
        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found. Use list_available_tools to see available tools."})
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON args: {e}"})
        try:
            result = await tool.ainvoke(args)
            return str(result)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    list_tool = StructuredTool.from_function(
        func=_list_available_tools,
        name="list_available_tools",
        description="List all extended simulation tools (beyond core tools) with their descriptions. Use this to discover tools not in your default set.",
    )

    call_tool = StructuredTool.from_function(
        coroutine=_call_extended_tool,
        func=lambda name, args_json="{}": None,  # sync placeholder
        name="call_extended_tool",
        description="Call an extended simulation tool by name. Args: name (str) - tool name from list_available_tools, args_json (str) - JSON string of tool arguments.",
    )

    result = core_tools + [list_tool, call_tool]
    logger.info(
        f"Progressive MCP tools: core={len(core_tools)}, extended={len(extended_tools)}, "
        f"total injected={len(result)}"
    )
    return result
