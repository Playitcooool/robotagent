from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import tools.AnalysisTool
from prompts import AnalysisAgentPrompt
import yaml

with open("config/config.yml", "r", encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

chatBot = ChatOpenAI(
    base_url=config["model_url"],
    model=config["llm"]["analysis"],
    api_key="no_need",
)
analysis_agent_tools = []
for func_name in tools.AnalysisTool.__all__:
    function = getattr(tools.AnalysisTool, func_name)
    analysis_agent_tools.append(function)
analysis_agent = create_agent(
    model=chatBot,
    tools=analysis_agent_tools,
    system_prompt=AnalysisAgentPrompt.SYSTEM_PROMPT,
)
