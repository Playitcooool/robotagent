from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import sum_two_num, product_two_num
from langchain.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import select_best_and_worst, judge_pair, trajectory_to_judge_text


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
    print(tools)
    chat = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="qwen3-4b-instruct-2507-mlx",
        api_key="no_need",
    )
    judge = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="no_need", model="")
    agent = create_agent(model=chat, tools=tools, system_prompt=SYSTEM_PROMPT)
    prompts = [
        "please calculate the sum of 1 and 2",
        "please calculate the product of 1 and 2",
        "please calculate the sum of 3 and 2",
        "please calculate the product of 5 and 2",
        "please calculate the sum of 7 and 2",
        "please calculate the product of 8 and 2",
    ]

    # Store all trajectories for all prompts
    all_trajectories = []
    dpo_pairs = []
    # Loop over each prompt
    for index, prompt in enumerate(prompts):
        print(f"Processing Prompt {index + 1}: {prompt}")

        prompt_trajectories = []  # ⭐ 存这个 prompt 的 4 条轨迹

        # Generate 4 different trajectories for each prompt
        for i in range(4):
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )

            messages = response["messages"]
            trajectory = []

            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    item = {
                        "role": (
                            "human"
                            if isinstance(msg, HumanMessage)
                            else "ai" if isinstance(msg, AIMessage) else "tool"
                        ),
                        "type": msg.__class__.__name__,
                        "content": msg.content,
                        "prompt_id": index,
                        "attempt_id": i,
                    }

                    if isinstance(msg, ToolMessage):
                        item["tool_name"] = msg.name
                        item["tool_call_id"] = msg.tool_call_id

                    trajectory.append(item)

            prompt_trajectories.append(trajectory)
            all_trajectories.append(trajectory)  # 如果你还想全量保存
        # ===== 用外部 LLM Judge 选 best / worst =====
    chosen, rejected = await select_best_and_worst(prompt, prompt_trajectories)

    dpo_pairs.append(
        {
            "prompt_id": index,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    # Write all trajectories to a JSON file
    with open("RL/trajectory.json", "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

    print("Trajectories saved successfully!")


asyncio.run(main())
