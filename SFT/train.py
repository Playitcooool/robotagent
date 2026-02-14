import argparse
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for student model.")
    parser.add_argument(
        "--data",
        type=str,
        default="trajectories.jsonl",
        help="Path to teacher trajectories JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Student model name or local path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sft_ckpt",
        help="Output directory for checkpoints.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
    )
    return parser.parse_args()


def format_messages(messages: List[Dict], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "tool":
            prefix = "Tool"
        else:
            prefix = "User"
        lines.append(f"### {prefix}:\n{content}")
    return "\n".join(lines)


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int, max_samples: int):
        self.samples: List[str] = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                messages = item.get("messages")
                if not messages:
                    continue
                text = format_messages(messages, tokenizer)
                self.samples.append(text)
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
    ratio = (trainable / total) * 100 if total else 0
    print(f"Trainable params: {trainable} / {total} ({ratio:.2f}%)")


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    target_modules = [m.strip() for m in args.lora_targets.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    dataset = SFTDataset(
        path=args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
