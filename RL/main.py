from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import sum_two_num, product_two_num
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from DPO import DPO


async def main():
    client = MultiServerMCPClient(
        {
            "pybullet": {
                "transport": "http",
                "url": "http://localhost:8001/mcp",
            }
        }
    )
    SYSTEM_PROMPT = (
        "You are a helpful assistant, please select proper tools to meet user's request"
    )
    mcp_tool = await client.get_tools()
    tools = [sum_two_num, product_two_num] + mcp_tool
    chat = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="qwen3-4b-instruct-2507-mlx",
        api_key="no_need",
    )
    judge = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="no_need",
        model="deepseek-r1-distill-qwen-1.5b@3bit",
    )
    agent = create_agent(model=chat, tools=tools, system_prompt=SYSTEM_PROMPT)
    prompts = [
        "please calculate the sum of 1 and 2",
        "please calculate the product of 1 and 2",
        "please calculate the sum of 3 and 2",
        "please calculate the product of 5 and 2",
        "please calculate the sum of 7 and 2",
        "please calculate the product of 8 and 2",
    ]
    dpo = DPO(
        model_path="/Volumes/Samsung/lmstudio/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
        prompts=prompts,
        judge=judge,
        agent=agent,
        samples_per_prompt=2,
    )
    await dpo.run()


asyncio.run(main())
