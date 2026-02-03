# offline_dpo_train.py
import json
import torch
import re
from datasets import Dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, ToolMessage

# ================= 1. 加载轨迹 =================
with open("trajectories.json", "r", encoding="utf-8") as f:
    trajectories = json.load(f)

# ================= 2. 初始化 judge 模型 =================
judge = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="qwen3-0.6b-mlx",
    api_key="no_need",
)


# ================= 3. 构建 DPO 数据 =================
async def build_dpo_pairs(trajectories, judge_model):
    """
    使用 judge 模型对每条轨迹评分，然后选择 chosen/rejected 构建 DPO pair
    trajectories: List[List[message]]  flat list，每条 trajectory 是消息字典列表
    judge_model: LangChain 模型实例
    """
    # ===== 1. 按 prompt_id 分组 =====
    prompt_groups = {}
    for traj in trajectories:
        if not traj:
            continue
        prompt_id = traj[0].get("prompt_id", None)
        if prompt_id is None:
            continue
        prompt_groups.setdefault(prompt_id, []).append(traj)

    dpo_pairs = []

    for prompt_id, traj_group in prompt_groups.items():
        if len(traj_group) < 2:
            # 至少两条轨迹才能构建 preference
            continue

        # ===== 2. 获取 prompt 文本（第一个 human 消息） =====
        prompt_text = None
        for msg in traj_group[0]:
            if msg["role"] == "human":
                prompt_text = msg["content"]
                break
        if prompt_text is None:
            prompt_text = f"prompt_{prompt_id}"

        # ===== 3. 将每条 trajectory 转为 judge 文本 =====
        traj_texts = []
        for traj in traj_group:
            text = ""
            for msg in traj:
                role = msg["role"]
                content = msg["content"]
                if role == "tool":
                    content = f"[TOOL {msg.get('tool_name','')}]: {content}"
                text += f"{role.upper()}: {content}\n"
            traj_texts.append(text)

        # ===== 4. 使用 judge 模型打分 =====
        scores = []
        for traj_text in traj_texts:
            judge_prompt = (
                f"Please score the quality of the following assistant trajectory from 0 (worst) to 10 (best):\n\n"
                f"{traj_text}"
            )
            try:
                # ✅ 这里直接用 HumanMessage 包装
                response = await judge_model.ainvoke(
                    [HumanMessage(content=judge_prompt)]
                )
                # ✅ AIMessage 对象直接用 .content
                score_str = response.content.strip()
                # 提取文本中的数字
                match = re.search(r"(\d+(\.\d+)?)", score_str)
                score = float(match.group(1)) if match else 0.0
            except Exception as e:
                print(f"Judge error for prompt {prompt_id}: {e}")
                score = 0.0
            scores.append(score)

        # ===== 5. 选出最高分和最低分 =====
        chosen_idx = int(scores.index(max(scores)))
        rejected_idx = int(scores.index(min(scores)))

        pair = {
            "prompt": prompt_text,
            "chosen": traj_texts[chosen_idx],
            "rejected": traj_texts[rejected_idx],
        }
        dpo_pairs.append(pair)

    return dpo_pairs


import asyncio

dpo_pairs = asyncio.run(build_dpo_pairs(trajectories, judge))

# ================= 4. 转为 HuggingFace Dataset =================
dataset = Dataset.from_list(dpo_pairs)

# ================= 5. Tokenizer =================
model_path = "/Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ================= 6. Policy & Reference Model =================
device = "mps" if torch.backends.mps.is_available() else "cuda"
policy_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)
policy_model.config.pad_token_id = tokenizer.eos_token_id

reference_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)
reference_model.config.pad_token_id = tokenizer.eos_token_id

# ================= 7. LoRA（推荐） =================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
policy_model = get_peft_model(policy_model, lora_config)

# ================= 8. DPO Config =================
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

# ================= 9. Trainer =================
trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    args=dpo_config,
    train_dataset=dataset,
)

# ================= 10. 开始训练 =================
trainer.train()

# ================= 11. 保存 =================
policy_model.save_pretrained("./dpo_ckpt/final")
tokenizer.save_pretrained("./dpo_ckpt/final")
print("✅ Offline DPO training finished")
