from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage, AIMessage, ToolMessage
import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录（tools目录）
current_dir = os.path.dirname(current_file)
# 获取项目根目录（tools的上一级目录）
root_dir = os.path.dirname(current_dir)
# 将根目录添加到Python的系统路径中
sys.path.append(root_dir)

from prompts import MainAgentPrompt
from tools.SubAgentTool import init_subagents


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
                print(msg)
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
    # ===== 初始化 ChatOpenAI agent =====
    subagents = list(await init_subagents())
    chat = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="lmstudio-community:qwen3-4b-instruct-2507-mlx",
        api_key="no_need",
    )
    agent = create_deep_agent(
        model=chat,
        tools=[],
        system_prompt=MainAgentPrompt.SYSTEM_PROMPT,
        subagents=subagents,
    )
    # ===== 采集 prompt 列表 =====
    prompts = []
    with open(
        "/Volumes/Samsung/Projects/robotagent/RL/data.txt", "r", encoding="utf-8"
    ) as f:
        for line in f.readlines():
            prompts.append(line)

    all_trajectories = await sample_trajectories(agent, prompts, samples_per_prompt=4)

    # ===== 保存 JSON =====
    with open("trajectories.json", "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    print("✅ Trajectories saved to trajectories.json")


asyncio.run(main())
