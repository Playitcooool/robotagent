import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI


@dataclass
class DeepAgentClient:
    base_url: str
    model: str
    api_key: str
    max_retries: int
    backoff_factor: float
    initial_delay: float
    temperature: float
    max_tokens: int

    async def chat(
        self,
        system_prompt: str,
        user_content: str,
    ) -> str:
        chat = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        agent = create_deep_agent(
            model=chat,
            tools=[],
            system_prompt=system_prompt,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=self.max_retries,
                    backoff_factor=self.backoff_factor,
                    initial_delay=self.initial_delay,
                )
            ],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]}
        )

        for msg in reversed(result.get("messages", [])):
            cls_name = msg.__class__.__name__
            if cls_name == "AIMessage":
                return str(msg.content).strip()
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        raise RuntimeError("No assistant response in deepagent result")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def build_score_prompt(traj: Dict[str, Any]) -> Dict[str, str]:
    rubric = (
        "你是严格评审。请根据任务完成度、正确性、清晰度、鲁棒性、冗余度对轨迹评分。"
        "返回严格 JSON，不要额外文本。"
    )
    user_content = {
        "instruction": "请评分并解释。",
        "trajectory": traj.get("messages", []),
        "output_schema": {
            "overall_score": "0-10 float",
            "task_completion": "0-10 float",
            "correctness": "0-10 float",
            "clarity": "0-10 float",
            "robustness": "0-10 float",
            "conciseness": "0-10 float",
            "pros": ["string"],
            "cons": ["string"],
            "brief_reason": "string",
        },
    }
    return {"system": rubric, "user": json.dumps(user_content, ensure_ascii=False)}


async def score_trajectories(
    client: DeepAgentClient,
    trajectory_path: str,
    score_path: str,
) -> None:
    trajectories = load_jsonl(trajectory_path)
    existing = load_jsonl(score_path)
    done_keys = {(x.get("prompt_id"), x.get("attempt_id")) for x in existing}

    for traj in trajectories:
        key = (traj.get("prompt_id"), traj.get("attempt_id"))
        if key in done_keys:
            continue

        prompt = build_score_prompt(traj)
        raw = await client.chat(prompt["system"], prompt["user"])
        parsed = extract_first_json_object(raw)
        if parsed is None:
            parsed = {
                "overall_score": 0.0,
                "task_completion": 0.0,
                "correctness": 0.0,
                "clarity": 0.0,
                "robustness": 0.0,
                "conciseness": 0.0,
                "pros": [],
                "cons": ["judge_output_not_json"],
                "brief_reason": raw[:300],
            }

        record: Dict[str, Any] = {
            "prompt_id": traj.get("prompt_id"),
            "attempt_id": traj.get("attempt_id"),
            "prompt": traj.get("prompt", ""),
            "response": traj.get("response", ""),
            "score": parsed,
            "meta": {
                "model": client.model,
                "created_at": int(time.time()),
                "method": "deep_agent",
                "judge_temperature": client.temperature,
            },
        }
        append_jsonl(score_path, record)
        print(
            f"[score] saved prompt={record['prompt_id']} attempt={record['attempt_id']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score trajectories only.")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument(
        "--model", type=str, default="lmstudio-community-qwen3-4b-instruct-2507-mlx"
    )
    parser.add_argument("--api_key", type=str, default="no_need")
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default="output/training_free_grpo/trajectories.jsonl",
    )
    parser.add_argument(
        "--score_path",
        type=str,
        default="output/training_free_grpo/trajectory_scores.jsonl",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    parser.add_argument("--initial_delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.score_path)
    if output_dir:
        ensure_dir(output_dir)

    client = DeepAgentClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        initial_delay=args.initial_delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    await score_trajectories(
        client=client,
        trajectory_path=args.trajectory_path,
        score_path=args.score_path,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
