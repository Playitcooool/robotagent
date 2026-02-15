import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


def build_summary_prompt(
    top: List[Dict[str, Any]], bottom: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    system = (
        "你是经验提炼器。基于高分/低分样本，提炼可执行策略。"
        "仅返回严格 JSON，不要额外文本。"
    )
    payload = {
        "task": "提炼可迁移经验",
        "top_examples": top,
        "bottom_examples": bottom,
        "output_schema": {
            "principles": ["string"],
            "dos": ["string"],
            "donts": ["string"],
            "failure_patterns": ["string"],
            "checklist": ["string"],
            "one_paragraph_summary": "string",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def pick_top_bottom(
    scores: List[Dict[str, Any]], top_k: int, bottom_k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def get_score(x: Dict[str, Any]) -> float:
        try:
            return float(x.get("score", {}).get("overall_score", 0.0))
        except Exception:
            return 0.0

    ranked = sorted(scores, key=get_score, reverse=True)
    top = ranked[: max(0, top_k)]
    bottom = ranked[-max(0, bottom_k) :] if bottom_k > 0 else []
    return top, bottom


def summarize_experience(
    client: ChatClient,
    score_path: str,
    memory_json_path: str,
    memory_md_path: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    bottom_k: int,
) -> None:
    scores = load_jsonl(score_path)
    if not scores:
        raise RuntimeError("No score data found. Run score step first.")

    top, bottom = pick_top_bottom(scores, top_k=top_k, bottom_k=bottom_k)
    raw = client.chat(
        build_summary_prompt(top=top, bottom=bottom),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = extract_first_json_object(raw)
    if parsed is None:
        parsed = {
            "principles": [],
            "dos": [],
            "donts": [],
            "failure_patterns": ["summary_output_not_json"],
            "checklist": [],
            "one_paragraph_summary": raw[:800],
        }

    external_memory = {
        "meta": {
            "model": client.model,
            "created_at": int(time.time()),
            "top_k": top_k,
            "bottom_k": bottom_k,
            "source_score_file": score_path,
            "method": "training_free_grpo",
        },
        "experience": parsed,
    }

    with open(memory_json_path, "w", encoding="utf-8") as f:
        json.dump(external_memory, f, ensure_ascii=False, indent=2)

    lines = [
        "# Training-Free GRPO External Memory",
        "",
        f"- model: {client.model}",
        f"- created_at: {external_memory['meta']['created_at']}",
        f"- top_k: {top_k}",
        f"- bottom_k: {bottom_k}",
        "",
        "## Summary",
        parsed.get("one_paragraph_summary", ""),
        "",
        "## Principles",
    ]
    for x in parsed.get("principles", []):
        lines.append(f"- {x}")
    lines.extend(["", "## Dos"])
    for x in parsed.get("dos", []):
        lines.append(f"- {x}")
    lines.extend(["", "## Donts"])
    for x in parsed.get("donts", []):
        lines.append(f"- {x}")
    lines.extend(["", "## Failure Patterns"])
    for x in parsed.get("failure_patterns", []):
        lines.append(f"- {x}")
    lines.extend(["", "## Checklist"])
    for x in parsed.get("checklist", []):
        lines.append(f"- {x}")

    with open(memory_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[summarize] external memory saved to {memory_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scored trajectories only.")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument(
        "--model", type=str, default="lmstudio-community-qwen3-4b-instruct-2507-mlx"
    )
    parser.add_argument("--api_key", type=str, default="no_need")
    parser.add_argument(
        "--score_path",
        type=str,
        default="output/training_free_grpo/trajectory_scores.jsonl",
    )
    parser.add_argument(
        "--memory_json_path",
        type=str,
        default="output/training_free_grpo/external_memory.json",
    )
    parser.add_argument(
        "--memory_md_path",
        type=str,
        default="output/training_free_grpo/external_memory.md",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bottom_k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_output_dir = os.path.dirname(args.memory_json_path)
    md_output_dir = os.path.dirname(args.memory_md_path)
    if json_output_dir:
        ensure_dir(json_output_dir)
    if md_output_dir:
        ensure_dir(md_output_dir)

    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )
    summarize_experience(
        client=client,
        score_path=args.score_path,
        memory_json_path=args.memory_json_path,
        memory_md_path=args.memory_md_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )


if __name__ == "__main__":
    main()
