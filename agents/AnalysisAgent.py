from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import tools.AnalysisTool
from prompts import AnalysisAgentPrompt

chatBot = ChatOpenAI(
    base_url="http://localhost:1234",
    model="qwen2.5.1-coder-7b-instruct",
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
