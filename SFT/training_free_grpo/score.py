import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str
    timeout: int = 120

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTPError {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Request failed: {e}") from e

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")

        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Invalid content in response: {data}")
        return content.strip()


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


def build_score_prompt(traj: Dict[str, Any]) -> List[Dict[str, str]]:
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
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def score_trajectories(
    client: ChatClient,
    trajectory_path: str,
    score_path: str,
    temperature: float,
    max_tokens: int,
) -> None:
    trajectories = load_jsonl(trajectory_path)
    existing = load_jsonl(score_path)
    done_keys = {(x.get("prompt_id"), x.get("attempt_id")) for x in existing}

    for traj in trajectories:
        key = (traj.get("prompt_id"), traj.get("attempt_id"))
        if key in done_keys:
            continue

        raw = client.chat(
            build_score_prompt(traj), temperature=temperature, max_tokens=max_tokens
        )
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
                "judge_temperature": temperature,
                "created_at": int(time.time()),
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.score_path)
    if output_dir:
        ensure_dir(output_dir)

    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )
    score_trajectories(
        client=client,
        trajectory_path=args.trajectory_path,
        score_path=args.score_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
