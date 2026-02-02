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
        model_path,
        agent,
        judge,
        prompts,
        samples_per_prompt=4,
        dpo_batch_size=None,
    ):
        self.model_path = model_path
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
        import torch
        from datasets import Dataset
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from trl import DPOTrainer, DPOConfig

        print(f"DPO update on {len(self.dpo_pairs)} preference pairs")

        # ========= 1. 构建 Dataset =========
        dataset = Dataset.from_list(self.dpo_pairs)

        # ========= 2. Tokenizer =========
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # ========= 3. Policy & Reference Model =========
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )

        reference_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )

        # ========= 4. LoRA（强烈建议） =========
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )

        policy_model = get_peft_model(policy_model, lora_config)

        # ========= 5. DPO Config =========
        dpo_config = DPOConfig(
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=1,
            output_dir="./dpo_ckpt",
            report_to="none",
        )

        # ========= 6. Trainer =========
        trainer = DPOTrainer(
            model=policy_model,
            ref_model=reference_model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # ========= 7. Train =========
        trainer.train()

        # ========= 8. 保存 =========
        policy_model.save_pretrained("./dpo_ckpt/final")
        tokenizer.save_pretrained("./dpo_ckpt/final")

        print("✅ DPO update finished")

    async def run(self, iterations=3):
        for it in range(iterations):
            print(f"\n===== Online RL Iteration {it} =====")
            await self.collect_one_batch()
            self.dpo_update()
