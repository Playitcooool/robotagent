from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.AnalysisTool import __all__

chatBot = ChatOllama(base_url="http://localhost:11434", model="qwen3:4b")
analysis_agent = create_agent(
    model=chatBot,
    tools=[__all__],
)
