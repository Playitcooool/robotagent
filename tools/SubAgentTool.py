from langchain_core.tools import tool
from agents.AnalysisAgent import analysis_agent


@tool(
    response_format="content",
    description="call analysis agent to analysis simulation results",
)
def call_analysis_agent(query: str):
    result = analysis_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool(
    response_format="content",
    description="call an agent to implement specific simulation task",
)
def call_simulation_agent(query: str):
    result = analysis_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


__all__ = [
    "call_simulation_agent",
    "call_analysis_agent",
]
