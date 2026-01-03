from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.general_tools import __all__
from tools.SubAgentTool import call_analysis_agent, call_simulation_agent

chatBot = ChatOllama(base_url="http://localhost:11434", model="qwen3:8b")
agent = create_agent(
    model=chatBot,
    tools=[__all__, call_simulation_agent, call_analysis_agent],
)
