import asyncio
import json
import os
import sys
from typing import Set, Tuple

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import ToolRetryMiddleware

# ===============================
# 路径处理（与你原来一致）
# ===============================
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from prompts import MainAgentPrompt
from tools.SubAgentTool import init_subagents

# ===============================
# 配置
# ===============================
PROMPT_FILE = "/Volumes/Samsung/Projects/robotagent/SFT/data.txt"
OUTPUT_JSONL = "trajectories.jsonl"
SAMPLES_PER_PROMPT = 1


# ===============================
# 工具函数：加载已完成的 (prompt_id, attempt_id)
# ===============================
def load_finished_keys(jsonl_path: str) -> Set[Tuple[int, int]]:
    finished = set()
    if not os.path.exists(jsonl_path):
        return finished

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                finished.add((item["prompt_id"], item["attempt_id"]))
            except Exception:
                # 防止单行损坏导致整体失败
                continue
    return finished


# ===============================
# 核心：增量式采样（可断点重启）
# ===============================
async def sample_trajectories_incremental(
    agent,
    prompts,
    samples_per_prompt: int,
    output_path: str,
):
    finished_keys = load_finished_keys(output_path)

    print(f"🔁 已检测到完成的 trajectory 数量: {len(finished_keys)}")

    for prompt_id, prompt in enumerate(prompts):
        print(f"\n📌 Prompt {prompt_id}: {prompt.strip()}")

        for attempt_id in range(samples_per_prompt):
            key = (prompt_id, attempt_id)

            if key in finished_keys:
                print(f"⏭ 跳过 Prompt {prompt_id} Attempt {attempt_id}")
                continue

            print(f"▶ 运行 Prompt {prompt_id} Attempt {attempt_id}")

            try:
                response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
            except Exception as e:
                print(f"❌ 调用失败，跳过该样本: {e}")
                continue

            messages = []

            for msg in response["messages"]:
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    if isinstance(msg, HumanMessage):
                        role = "user"
                    elif isinstance(msg, AIMessage):
                        role = "assistant"
                    else:
                        role = "tool"

                    item = {
                        "role": role,
                        "content": msg.content,
                    }

                    if isinstance(msg, ToolMessage):
                        if msg.name:
                            item["name"] = msg.name
                        if msg.tool_call_id:
                            item["tool_call_id"] = msg.tool_call_id

                    messages.append(item)

            record = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "messages": messages,
            }

            # ✅ 立刻追加写入（原子性强）
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            finished_keys.add(key)
            print(f"✅ 已保存 Prompt {prompt_id} Attempt {attempt_id}")


# ===============================
# 主入口
# ===============================
async def main():
    # 初始化 subagents
    subagents = list(await init_subagents())

    # 初始化 LLM
    chat = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="qwen-qwen3-14b-mlx@4bit",
        api_key="no_need",
    )

    # 创建 Deep Agent
    agent = create_deep_agent(
        model=chat,
        tools=[],
        system_prompt=MainAgentPrompt.SYSTEM_PROMPT,
        subagents=subagents,
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ],
    )

    # 读取 prompts
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"📄 加载 prompt 数量: {len(prompts)}")

    await sample_trajectories_incremental(
        agent=agent,
        prompts=prompts,
        samples_per_prompt=SAMPLES_PER_PROMPT,
        output_path=OUTPUT_JSONL,
    )

    print("\n🎉 采样完成（或已全部完成）")


if __name__ == "__main__":
    asyncio.run(main())
