import sys
import os
import asyncio

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录（tools目录）
current_dir = os.path.dirname(current_file)
# 获取项目根目录（tools的上一级目录）
root_dir = os.path.dirname(current_dir)
# 将根目录添加到Python的系统路径中
sys.path.append(root_dir)

# 现在可以直接使用绝对导入
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import yaml
import logging
from tools import AnalysisTool
from deepagents import CompiledSubAgent

# 替换相对导入为绝对导入
from prompts import AnalysisAgentPrompt, SimulationAgentPrompt
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("uvicorn")

with open(
    "/Volumes/Samsung/Projects/robotagent/config/config.yml", "r", encoding="utf-8"
) as f:
    config = yaml.load(f.read(), yaml.FullLoader)

# ---------------------------------------------------------------------------
# Cached MCP tools and model clients (expensive to initialize)
# ---------------------------------------------------------------------------

_cached_mcp_tools: list | None = None
_cached_analysis_tools: list | None = None
_cached_analysis_chat: "ChatOpenAI | None" = None
_cached_simulation_chat: "ChatOpenAI | None" = None
_mcp_tools_lock = asyncio.Lock()


def _resolve_mcp_service_urls() -> dict[str, str]:
    """Resolve MCP endpoints from config with backward compatibility."""
    mcp_cfg = config.get("mcp") or {}

    # Preferred format:
    # mcp:
    #   servers:
    #     pybullet: "http://127.0.0.1:8001/mcp"
    #     gazebo: "http://127.0.0.1:8002/mcp"
    servers = mcp_cfg.get("servers")
    if isinstance(servers, dict) and servers:
        resolved = {}
        for name, raw_url in servers.items():
            if not raw_url:
                continue
            url = str(raw_url).strip()
            if not url:
                continue
            if not url.endswith("/mcp"):
                url = url.rstrip("/") + "/mcp"
            resolved[str(name).strip()] = url
        if resolved:
            return resolved

    # Backward compatible format:
    # mcp:
    #   ip: "http://localhost"
    #   port: "8001"
    # Optional:
    #   gazebo_port: "8002"
    base = str(mcp_cfg.get("ip") or "http://127.0.0.1").rstrip("/")
    pybullet_port = str(mcp_cfg.get("port") or mcp_cfg.get("pybullet_port") or "8001")
    gazebo_port = mcp_cfg.get("gazebo_port")

    urls = {
        "pybullet": f"{base}:{pybullet_port}/mcp",
    }
    if gazebo_port:
        urls["gazebo"] = f"{base}:{str(gazebo_port)}/mcp"
    return urls


def reset_cached_mcp_tools() -> None:
    """Reset cached MCP tools and simulation chat. Call on stream errors to force reconnect."""
    global _cached_mcp_tools, _cached_simulation_chat
    _cached_mcp_tools = None
    _cached_simulation_chat = None


async def init_subagents(
    experiences: list | None = None,
    max_experiences_in_subagent: int = 10,
):
    """
    Initialize subagents with optional experience injection.

    MCP tools and model clients are cached globally (expensive to reconnect).
    Agent graphs are rebuilt every call with the current experiences (cheap).

    Args:
        experiences: List of experience dicts to inject into subagent system prompts.
        max_experiences_in_subagent: Max experiences to inject per subagent.
    """
    subagents = []
    experiences = experiences or []

    # Build experience context for subagents
    def _build_exp_context(exps: list) -> str:
        if not exps:
            return ""
        lines = ["", "【历史经验（请遵守）】"]
        for exp in exps[-max_experiences_in_subagent:]:
            lines.append(f"- prompt_id={exp.get('prompt_id')}, score={exp.get('score', 0.0)}")
            s = str(exp.get("summary", "")).strip()
            if s:
                lines.append(f"  总结: {s}")
            for p in exp.get("principles", [])[:3]:
                lines.append(f"  原则: {p}")
        return "\n".join(lines)

    # Use cached analysis tools/client if available
    global _cached_analysis_tools, _cached_analysis_chat
    if _cached_analysis_tools is None:
        _cached_analysis_tools = []
        for func_name in AnalysisTool.__all__:
            function = getattr(AnalysisTool, func_name)
            _cached_analysis_tools.append(function)
    if _cached_analysis_chat is None:
        _cached_analysis_chat = ChatOpenAI(
            base_url=config.get("analysis_model_url", config["model_url"]),
            model=config.get("analysis_llm", config["llm"]),
            api_key=config.get("analysis_api_key", config.get("api_key", "no_need")),
        )

    # Rebuild analysis agent graph with current experiences
    analysis_system = AnalysisAgentPrompt.SYSTEM_PROMPT + _build_exp_context(experiences)
    analysis_graph = create_agent(
        model=_cached_analysis_chat,
        tools=_cached_analysis_tools,
        system_prompt=analysis_system,
    )
    analysis_agent = CompiledSubAgent(
        name="data-analyzer",
        description="Specialized agent for complex data analysis tasks",
        runnable=analysis_graph,
    )
    subagents.append(analysis_agent)

    disable_sim = str(os.environ.get("DISABLE_SIM_SUBAGENT", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if disable_sim:
        logger.info("Simulation subagent disabled by DISABLE_SIM_SUBAGENT")
        return tuple(subagents)

    # Cache MCP tools (expensive, loaded once)
    global _cached_mcp_tools, _cached_simulation_chat
    if _cached_mcp_tools is None:
        max_retries = int(os.environ.get("SIM_MCP_MAX_RETRIES", "8"))
        retry_delay_s = float(os.environ.get("SIM_MCP_RETRY_DELAY_SECONDS", "1.0"))
        service_urls = _resolve_mcp_service_urls()
        logger.info(f"MCP service endpoints: {service_urls}")
        _cached_mcp_tools = []
        loaded_services = []
        errors = {}

        for service_name, service_url in service_urls.items():
            service_tools = None
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    client = MultiServerMCPClient(
                        {
                            service_name: {
                                "transport": "http",
                                "url": service_url,
                            }
                        }
                    )
                    # 顺序加载每个服务的工具，避免并发gather导致的unhandled异常
                    service_tools = await client.get_tools(server_name=service_name)
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{service_name} MCP not ready (attempt {attempt}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay_s)

            if service_tools:
                _cached_mcp_tools.extend(service_tools)
                loaded_services.append(service_name)
            else:
                errors[service_name] = str(last_error)

        if not _cached_mcp_tools:
            raise RuntimeError(f"Simulation MCP unavailable: {errors}")
        logger.info(
            f"MCP tools loaded from services={loaded_services}, total={len(_cached_mcp_tools)}"
        )

    if _cached_simulation_chat is None:
        _cached_simulation_chat = ChatOpenAI(
            base_url=config.get("simulation_model_url", config["model_url"]),
            model=config.get("simulation_llm", config["llm"]),
            api_key=config.get("simulation_api_key", config.get("api_key", "no_need")),
        )

    # Rebuild simulation agent graph with current experiences
    sim_system = SimulationAgentPrompt.SYSTEM_PROMPT + _build_exp_context(experiences)
    simulation_graph = create_agent(
        model=_cached_simulation_chat,
        tools=_cached_mcp_tools,
        system_prompt=sim_system,
    )
    simulation_agent = CompiledSubAgent(
        name="simulator",
        description="Specialized agent for executing PyBullet and Gazebo simulations",
        runnable=simulation_graph,
    )
    subagents.append(simulation_agent)

    return tuple(subagents)
