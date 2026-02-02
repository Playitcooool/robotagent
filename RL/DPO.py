import json
from langchain.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from utils import select_best_and_worst, trajectory_to_judge_text, judge_pair


class DPO:
    def __init__(
        self,
        agent,
        judge,
        prompts,
        samples_per_prompt=4,
        dpo_batch_size=None,
    ):
        self.agent = agent
        self.judge = judge
        self.prompts = prompts
        self.samples_per_prompt = samples_per_prompt
        self.dpo_batch_size = dpo_batch_size or len(prompts)

        self.all_trajectories = []
        self.dpo_pairs = []

    async def sample_trajectories(self, prompt, prompt_id):
        trajectories = []

        for i in range(self.samples_per_prompt):
            response = await self.agent.ainvoke(
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
                        "prompt_id": prompt_id,
                        "attempt_id": i,
                    }

                    if isinstance(msg, ToolMessage):
                        item["tool_name"] = msg.name
                        item["tool_call_id"] = msg.tool_call_id

                    trajectory.append(item)

            trajectories.append(trajectory)
            self.all_trajectories.append(trajectory)

        return trajectories

    async def judge_and_build_pair(self, prompt, prompt_id, trajectories):
        chosen, rejected = await select_best_and_worst(
            prompt=prompt,
            trajectories=trajectories,
            judge=self.judge,
        )

        pair = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        self.dpo_pairs.append(pair)
        return pair

    async def collect_one_batch(self):
        print("=== Collecting one online batch ===")
        self.dpo_pairs = []

        for idx, prompt in enumerate(self.prompts):
            trajectories = await self.sample_trajectories(prompt, idx)
            await self.judge_and_build_pair(prompt, idx, trajectories)

        return self.dpo_pairs

    def dpo_update(self):
        """
        这里之后你可以：
        - 导出到 JSON
        - 喂给 TRL DPOTrainer
        - 或自己算 logprob
        """
        print(f"DPO update on {len(self.dpo_pairs)} preference pairs")

        # 示例：先落盘
        with open("dpo_pairs.json", "w", encoding="utf-8") as f:
            json.dump(self.dpo_pairs, f, ensure_ascii=False, indent=2)

    async def run(self, iterations=3):
        for it in range(iterations):
            print(f"\n===== Online RL Iteration {it} =====")
            await self.collect_one_batch()
            self.dpo_update()
