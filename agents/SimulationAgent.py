from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import tools.SubAgentTool
from prompts import SimulationAgentPrompt
import yaml

with open("config/config.yml", "r", encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

chatBot = ChatOpenAI(
    base_url=config["model_url"],
    model=config["llm"]["simulation"],
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
