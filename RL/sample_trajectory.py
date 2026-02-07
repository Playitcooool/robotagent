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
        "初始化模拟环境，并在环境中添加一个物体并放置在 [0.2, 0.0, 0.02] 位置。",
        "从 [0.2, 0.0, 0.02] 开始，逐步将物体推向 [0.5, 0.5, 0.02]，记录每个步骤的状态。",
        "在 [0.4, 0.4, 0.02] 位置放置一个立方体，并用机器人臂从 [0.2, 0.0, 0.02] 抓取它，最终放置到 [0.6, 0.6, 0.02]。",
        "模拟物体从位置 [0.2, 0.0, 0.02] 推到 [0.6, 0.6, 0.02]，每次执行一个小步进，记录轨迹。",
        "请模拟从 [0.2, 0.0, 0.02] 到 [0.4, 0.4, 0.02] 的路径，记录机器人每步的动作。",
        "我没有给出摩擦力，请默认使用摩擦系数为 0.5，弹性系数为 0.9，生成物体的动态轨迹。",
        "在模拟中初始化一个物理环境，逐步调整摩擦力和弹性，并记录所有变化。",
        "创建机器人臂并从起点 [0.2, 0.0, 0.02] 移动到目标位置 [0.5, 0.5, 0.02]，请返回每一步的轨迹。",
        "从 [0.2, 0.0, 0.02] 启动物体推送动作，目标位置为 [0.6, 0.6, 0.02]，记录轨迹。",
        "生成一个机器臂抓取并放置物体的操作，将物体从 [0.2, 0.0, 0.02] 移动到 [0.5, 0.5, 0.02]，并返回每一步。",
        "请在模拟环境中加入两个物体，分别位于 [0.1, 0.0, 0.02] 和 [0.4, 0.4, 0.02]，并逐步将它们移动到 [0.6, 0.6, 0.02]。",
        "模拟机器人臂的路径规划，假设起始位置为 [0.2, 0.0, 0.02]，目标位置为 [0.5, 0.5, 0.02]，返回路径。",
        "调整摩擦系数为 0.3，弹性系数为 0.8，模拟物体的滚动路径并记录。",
        "抓取多个物体并逐步移动它们到 [0.6, 0.6, 0.02]，每次执行一个操作并记录每个步骤。",
        "模拟机器臂在路径规划中，按照给定的起始位置 [0.2, 0.0, 0.02] 和目标位置 [0.4, 0.4, 0.02]，每个步骤的路径。",
        "将多个物体抓取并分别移动到目标位置 [0.5, 0.5, 0.02]，记录每个物体的位置变化。",
        "模拟从 [0.2, 0.0, 0.02] 开始推送物体，并逐步将它移动到 [0.5, 0.5, 0.02]，记录轨迹。",
        "在环境中加入一个物体并设定摩擦力为 0.6，生成物体沿着平面移动的轨迹。",
        "给定目标位置 [0.4, 0.4, 0.02]，请模拟路径规划并返回每个轨迹点。",
        "在没有给定目标位置时，创建一个简单的推送操作并记录每个轨迹点。",
        "请模拟一个机器人臂抓取并放置物体的动作，目标位置为 [0.5, 0.5, 0.02]，并记录路径。",
        "用机器人臂抓取物体并将其从 [0.2, 0.0, 0.02] 移动到目标位置 [0.4, 0.4, 0.02]，每个步骤执行。",
        "在没有明确给定摩擦力和弹性的情况下，使用默认的摩擦力（0.5）和弹性（0.9），模拟一个物体的移动。",
        "在模拟环境中初始化两个物体并分别设置它们的位置，然后开始推送动作。",
        "将一个物体从 [0.2, 0.0, 0.02] 推动到 [0.6, 0.6, 0.02]，每次执行一个小步进并记录轨迹。",
        "从 [0.2, 0.0, 0.02] 抓取物体并逐步将其移动到 [0.4, 0.4, 0.02]，记录轨迹。",
        "创建一个摩擦力为 0.8，弹性为 0.7 的物体，记录它在环境中的动态轨迹。",
        "给定起始位置和目标位置，模拟机器人臂路径规划并返回每一步的动作。",
        "没有明确给定摩擦力，请使用默认摩擦力 0.5 和弹性 0.9，模拟物体的移动。",
        "在模拟中，调整多个物体的摩擦力和弹性，并逐步记录每个操作的轨迹。",
        "通过视觉传感器捕捉并返回图像数据，模拟不同物体的位置变化。",
        "根据提供的目标位置 [0.4, 0.4, 0.02]，生成机器人臂的路径规划，并记录每个轨迹点。",
        "创建一个物体，初始位置 [0.2, 0.0, 0.02]，目标位置未给定，请使用默认目标。",
        "模拟机器人臂路径规划并返回每个步骤的动作，起始位置为 [0.2, 0.0, 0.02]。",
        "创建一个从起始位置 [0.2, 0.0, 0.02] 到目标位置 [0.4, 0.4, 0.02] 的路径规划。",
        "在没有给出目标位置时，请将物体从当前状态移动到默认目标位置 [0.6, 0.6, 0.02]。",
        "模拟物体推送路径，使用默认的摩擦力和弹性参数，记录每个轨迹点。",
        "模拟物体从 [0.2, 0.0, 0.02] 移动到目标位置 [0.4, 0.4, 0.02]，并逐步显示每个动作。",
        "创建一个机器人臂并规划路径，从 [0.2, 0.0, 0.02] 到 [0.5, 0.5, 0.02]，记录路径。",
        "根据目标位置 [0.5, 0.5, 0.02]，生成机器人臂的路径规划。",
        "通过机器人臂抓取物体并将其放置到目标位置，逐步显示每个路径点。",
        "没有给定目标位置时，使用默认位置 [0.4, 0.4, 0.02]，模拟路径规划。",
        "模拟机器人抓取动作，将物体从 [0.2, 0.0, 0.02] 放置到目标位置 [0.5, 0.5, 0.02]，逐步执行。",
        "创建一个物体，初始位置 [0.2, 0.0, 0.02]，并逐步将其移动到 [0.6, 0.6, 0.02]。",
        "模拟多个物体从不同位置同时被抓取并放置到目标位置 [0.6, 0.6, 0.02]，记录每个物体的轨迹。",
        "通过视觉传感器捕捉图像并返回，模拟物体移动的轨迹。",
        "没有明确给定摩擦力时，请使用默认摩擦力 0.5 和弹性 0.9，模拟物体的移动。",
        "从 [0.2, 0.0, 0.02] 启动推送操作并记录每步轨迹，目标位置 [0.6, 0.6, 0.02]。",
        "模拟一个物体的弹跳过程，初始摩擦力为 0.3，弹性为 0.8，返回每一步的轨迹。",
        "创建物理环境，模拟一个物体的路径规划，并逐步显示每个轨迹。",
        "模拟机器人臂从起始位置 [0.2, 0.0, 0.02] 到目标位置 [0.4, 0.4, 0.02]，每步显示。",
        "在没有给定弹性系数时，使用默认的弹性 0.9，模拟一个物体的轨迹。",
        "通过视觉传感器模拟环境，捕捉并返回图像数据。",
        "模拟一个立方体在环境中的移动，目标位置为 [0.5, 0.5, 0.02]，逐步生成轨迹。",
    ]

    all_trajectories = await sample_trajectories(agent, prompts, samples_per_prompt=4)

    # ===== 保存 JSON =====
    with open("trajectories.json", "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    print("✅ Trajectories saved to trajectories.json")


asyncio.run(main())
