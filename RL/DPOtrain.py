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

with open("dpo_pairs.json", "r", encoding="utf-8") as f:
    dpo_pairs = json.load(f)
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
