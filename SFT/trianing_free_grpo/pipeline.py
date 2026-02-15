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

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Invalid content in response: {data}")
        return content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-Free GRPO pipeline: collect trajectories, score, summarize, and export memory."
    )
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument(
        "--model", type=str, default="lmstudio-community-qwen3-4b-instruct-2507-mlx"
    )
    parser.add_argument("--api_key", type=str, default="no_need")
    parser.add_argument("--prompts", type=str, default="SFT/data.txt")
    parser.add_argument("--output_dir", type=str, default="output/trianing_free_grpo")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--samples_per_prompt", type=int, default=3)
    parser.add_argument("--collect_temperature", type=float, default=0.9)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--summary_temperature", type=float, default=0.2)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bottom_k", type=int, default=10)
    parser.add_argument(
        "--steps",
        type=str,
        default="collect,score,summarize",
        help="Comma separated: collect,score,summarize",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompts(path: str, max_prompts: int) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            prompts.append(text)
            if max_prompts > 0 and len(prompts) >= max_prompts:
                break
    return prompts


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


def collect_trajectories(
    client: ChatClient,
    prompts: List[str],
    output_path: str,
    system_prompt: str,
    samples_per_prompt: int,
    temperature: float,
    max_tokens: int,
) -> None:
    existing = load_jsonl(output_path)
    done_keys = {(x.get("prompt_id"), x.get("attempt_id")) for x in existing}

    for prompt_id, prompt in enumerate(prompts):
        for attempt_id in range(samples_per_prompt):
            key = (prompt_id, attempt_id)
            if key in done_keys:
                continue

            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            start_ts = time.time()
            response = client.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            elapsed = time.time() - start_ts

            record = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "prompt": prompt,
                "messages": messages + [{"role": "assistant", "content": response}],
                "response": response,
                "meta": {
                    "model": client.model,
                    "collect_temperature": temperature,
                    "elapsed_sec": round(elapsed, 3),
                    "created_at": int(time.time()),
                },
            }
            append_jsonl(output_path, record)
            print(f"[collect] saved prompt={prompt_id} attempt={attempt_id}")


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

        record = {
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


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    trajectory_path = os.path.join(args.output_dir, "trajectories.jsonl")
    score_path = os.path.join(args.output_dir, "trajectory_scores.jsonl")
    memory_json_path = os.path.join(args.output_dir, "external_memory.json")
    memory_md_path = os.path.join(args.output_dir, "external_memory.md")

    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )

    steps = {x.strip() for x in args.steps.split(",") if x.strip()}

    if "collect" in steps:
        prompts = load_prompts(args.prompts, args.max_prompts)
        collect_trajectories(
            client=client,
            prompts=prompts,
            output_path=trajectory_path,
            system_prompt=args.system_prompt,
            samples_per_prompt=args.samples_per_prompt,
            temperature=args.collect_temperature,
            max_tokens=args.max_tokens,
        )

    if "score" in steps:
        score_trajectories(
            client=client,
            trajectory_path=trajectory_path,
            score_path=score_path,
            temperature=args.judge_temperature,
            max_tokens=args.max_tokens,
        )

    if "summarize" in steps:
        summarize_experience(
            client=client,
            score_path=score_path,
            memory_json_path=memory_json_path,
            memory_md_path=memory_md_path,
            temperature=args.summary_temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
        )


if __name__ == "__main__":
    main()
