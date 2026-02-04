from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import yaml
import AnalysisTool
from prompts import AnalysisAgentPrompt, SimulationAgentPrompt
from langchain_mcp_adapters.client import MultiServerMCPClient

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f.read(), yaml.FullLoader)


@tool(
    response_format="content",
    description="call analysis agent to analysis simulation results",
)
def call_analysis_agent(query: str):
    chat = ChatOpenAI(
        base_url=config["model_url"], model=config["llm"]["analysis"], api_key="no_need"
    )
    analysis_tool = []
    for func_name in AnalysisTool.__all__:
        function = getattr(AnalysisTool, func_name)
        analysis_tool.append(function)
    analysis_agent = create_agent(
        model=chat, tool=analysis_tool, system_prompt=AnalysisAgentPrompt.SYSTEM_PROMPT
    )
    result = analysis_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool(
    response_format="content",
    description="call an agent to implement specific simulation task",
)
async def call_simulation_agent(query: str):
    client = MultiServerMCPClient(
        {"pybullet": {"transport": "http", "url": "http://localhost:8001/mcp"}}
    )
    chat = ChatOpenAI(
        base_url=config["model_url"],
        model=config["llm"]["simulation"],
        api_key="no_need",
    )
    simulation_tools = await client.get_tools()
    simulation_agent = create_agent(
        model=chat,
        tools=simulation_tools,
        system_prompt=SimulationAgentPrompt.SYSTEM_PROMPT,
    )
    result = await simulation_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


__all__ = [
    "call_simulation_agent",
    "call_analysis_agent",
]
