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
    prompts = [
        # ===== stacking（多约束堆叠）=====
        "Using PyBullet, stack three cubes of different sizes in ascending order of size on a table. The stack must remain stable for at least 100 simulation steps without external support.",
        "Stack two cubes while avoiding a cylindrical obstacle placed between the start positions and the target stacking location.",
        "Stack two cubes such that their center of mass is aligned within a tolerance of 0.01 along the x-axis.",
        # ===== grab_and_place（精度 + 路径约束）=====
        "Grab a cube from position [0.2, 0.0, 0.02] and place it at [0.4, 0.4, 0.02] while following a smooth, collision-free trajectory and minimizing end-effector velocity at placement.",
        "Pick up a cube from [0.1, -0.1, 0.02] and place it on a platform elevated at height 0.15, without exceeding a gripper force of 5N.",
        # ===== push_cube（路径 + 力控制）=====
        "Push a cube from [0.0, 0.0, 0.02] to [0.3, 0.3, 0.02] using only lateral contact, ensuring the cube does not rotate more than 10 degrees during the motion.",
        "Push a cube along a straight line of length 0.4 while avoiding a static obstacle placed at [0.2, 0.0, 0.02].",
        # ===== path_tracking（长时序 + 稳定性）=====
        "Move a sphere along a circular trajectory of radius 0.3 for 200 simulation steps while maintaining a constant speed and minimizing deviation from the path.",
        "Control a sphere to follow a figure-eight trajectory on the x-y plane for 150 steps without leaving the plane.",
        # ===== pick_and_throw（动力学 + 时序）=====
        "Pick up a cube from [0.0, 0.0, 0.02], lift it to height 0.2, and throw it such that it lands within 0.05 distance of target [0.5, 0.3, 0.02].",
        "Pick and throw a cube over a wall obstacle of height 0.15, ensuring the cube clears the wall and lands within a target zone.",
        # ===== multi-object reasoning（组合任务）=====
        "Pick up two cubes one by one and place them at target locations [0.3, 0.2, 0.02] and [0.4, -0.2, 0.02], ensuring the first cube is not disturbed while placing the second.",
        "Rearrange three cubes from a random initial configuration into a straight line with equal spacing of 0.1 between adjacent cubes.",
        # ===== failure-sensitive / RL-friendly =====
        "Using PyBullet, manipulate a cube to reach a target location while recovering from a failed grasp attempt without resetting the environment.",
        "Complete a stacking task where a small amount of random noise is added to the action at each timestep, and ensure the final configuration is stable.",
        # ===== long-horizon reasoning =====
        "Plan and execute a sequence of actions to clear a workspace by moving obstructing objects out of the way before stacking two cubes at the center.",
    ]

    all_trajectories = await sample_trajectories(agent, prompts, samples_per_prompt=4)

    # ===== 保存 JSON =====
    with open("trajectories.json", "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    print("✅ Trajectories saved to trajectories.json")


asyncio.run(main())
