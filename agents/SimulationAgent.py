from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import tools.SubAgentTool
from prompts import SimulationAgentPrompt

chatBot = ChatOpenAI(
    base_url="http://localhost:1234",
    model="qwen2.5.1-coder-7b-instruct",
    api_key="no_need",
)
simulation_agent_tools = []
for func_name in tools.SubAgentTool.__all__:
    function = getattr(tools.SubAgentTool, func_name)
    simulation_agent_tools.append(function)

simulation_agent = create_agent(
    model=chatBot,
    tools=simulation_agent_tools,
    system_prompt=SimulationAgentPrompt.SYSTEM_PROMPT,
)
