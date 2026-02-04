from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage, AIMessage, ToolMessage


async def sample_trajectories(agent, prompts, samples_per_prompt=4):
    all_trajectories = []

    for idx, prompt in enumerate(prompts):
        print(f"Processing Prompt {idx+1}: {prompt}")

        for attempt in range(samples_per_prompt):
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            trajectory = []

            for msg in response["messages"]:
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    item = {
                        "role": (
                            "human"
                            if isinstance(msg, HumanMessage)
                            else "ai" if isinstance(msg, AIMessage) else "tool"
                        ),
                        "type": msg.__class__.__name__,
                        "content": msg.content,
                        "prompt_id": idx,
                        "attempt_id": attempt,
                    }
                    if isinstance(msg, ToolMessage):
                        item["tool_name"] = msg.name
                        item["tool_call_id"] = msg.tool_call_id
                    trajectory.append(item)

            all_trajectories.append(trajectory)

    return all_trajectories


async def main():
    # ===== MCP 客户端获取工具 =====
    client = MultiServerMCPClient(
        {
            "pybullet": {
                "transport": "http",
                "url": "http://localhost:8001/mcp",
            }
        }
    )
    mcp_tool = await client.get_tools()
    tools = mcp_tool

    # ===== 初始化 ChatOpenAI agent =====
    SYSTEM_PROMPT = (
        "You are a helpful assistant, please select proper tools to meet user's request"
    )
    chat = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="lmstudio-community:qwen3-4b-instruct-2507-mlx",
        api_key="no_need",
    )
    agent = create_agent(model=chat, tools=tools, system_prompt=SYSTEM_PROMPT)

    # ===== 采集 prompt 列表 =====
    prompts = [
        # stacking
        "Stack two small cubes on top of each other using PyBullet simulation.",
        # grab_and_place
        "Grab a cube from position [0.2, 0.0, 0.02] and place it at [0.4, 0.4, 0.02].",
        "Grab a cube from position [0.1, -0.1, 0.02] and place it at [0.3, 0.2, 0.02].",
        # path_tracking
        "Move a sphere along a circular path of radius 0.3 for 120 steps.",
        "Move a sphere along a circular path of radius 0.5 for 60 steps.",
        # push_cube
        "Push a cube from [0.0, 0.0, 0.02] along the vector [0.2, 0.0, 0.0] in PyBullet.",
        "Push a cube from [0.1, 0.1, 0.02] along the vector [0.0, 0.3, 0.0].",
        # pick_and_throw
        "Pick a cube from [0.0, 0.0, 0.02], lift it, and throw it along the vector [0.3, 0.3, 0.2].",
        "Pick a cube from [0.2, 0.0, 0.02], lift it, and throw it along the vector [0.2, 0.4, 0.1].",
        # stacking variation
        "Stack two small cubes at different positions in PyBullet simulation.",
    ]

    all_trajectories = await sample_trajectories(agent, prompts, samples_per_prompt=4)

    # ===== 保存 JSON =====
    with open("trajectories.json", "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    print("✅ Trajectories saved to trajectories.json")


asyncio.run(main())
