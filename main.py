from langchain_ollama import ChatOllama
from langchain.agents import create_agent

chatBot = ChatOllama(base_url="http://localhost:11434", model="qwen3:8b")
agent = create_agent(
    model=chatBot,
    tools=[],
)
