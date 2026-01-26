from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

chatBot = ChatOpenAI(
    base_url="http://localhost:1234",
    model="qwen2.5.1-coder-7b-instruct",
    api_key="no_need",
)
agent = create_agent(
    model=chatBot,
    tools=[],
)
