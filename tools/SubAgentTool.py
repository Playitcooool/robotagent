import sys
import os

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


async def init_subagents():
    subagents = []

    analysis_chat = ChatOpenAI(
        base_url=config["model_url"], model=config["llm"]["analysis"], api_key="no_need"
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
        sim_client = MultiServerMCPClient(
            {
                "pybullet": {"transport": "http", "url": "http://127.0.0.1:8001/mcp"},
                "gazebo": {"transport": "http", "url": "http://127.0.0.1:8002/mcp"},
            }
        )
        simulation_chat = ChatOpenAI(
            base_url=config["model_url"],
            model=config["llm"]["simulation"],
            api_key="no_need",
        )
        simulation_tools = await sim_client.get_tools()
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
