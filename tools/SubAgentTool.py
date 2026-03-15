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


async def init_subagents():
    subagents = []

    analysis_chat = ChatOpenAI(
        base_url=config["model_url"],
        model=config["llm"],
        api_key=config.get("api_key", "no_need"),
    )
    analysis_tool = []
    for func_name in AnalysisTool.__all__:
        function = getattr(AnalysisTool, func_name)
        analysis_tool.append(function)
    analysis_graph = create_agent(
        model=analysis_chat,
        tools=analysis_tool,
        system_prompt=AnalysisAgentPrompt.SYSTEM_PROMPT,
    )

    analysis_agent = CompiledSubAgent(
        name="data-analyzer",
        description="Specialized agent for complex data analysis tasks",
        runnable=analysis_graph,
    )
    subagents.append(analysis_agent)

    try:
        max_retries = int(os.environ.get("SIM_MCP_MAX_RETRIES", "8"))
        retry_delay_s = float(os.environ.get("SIM_MCP_RETRY_DELAY_SECONDS", "1.0"))
        service_urls = _resolve_mcp_service_urls()
        logger.info(f"MCP service endpoints: {service_urls}")
        simulation_tools = []
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
                    service_tools = await client.get_tools()
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{service_name} MCP not ready (attempt {attempt}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay_s)

            if service_tools:
                simulation_tools.extend(service_tools)
                loaded_services.append(service_name)
            else:
                errors[service_name] = str(last_error)

        if not simulation_tools:
            raise RuntimeError(
                f"Simulation MCP unavailable: {errors}"
            )

        logger.info(
            f"Simulation tools loaded from services={loaded_services}, total={len(simulation_tools)}"
        )

        simulation_chat = ChatOpenAI(
            base_url=config["model_url"],
            model=config["llm"],
            api_key=config.get("api_key", "no_need"),
        )
        simulation_graph = create_agent(
            model=simulation_chat,
            tools=simulation_tools,
            system_prompt=SimulationAgentPrompt.SYSTEM_PROMPT,
        )
        simulation_agent = CompiledSubAgent(
            name="simulator",
            description="Specialized agent for executing PyBullet and Gazebo simulations",
            runnable=simulation_graph,
        )
        subagents.append(simulation_agent)
    except Exception as e:
        logger.warning(f"Simulation subagent unavailable, continue without it: {str(e)}")

    return tuple(subagents)
