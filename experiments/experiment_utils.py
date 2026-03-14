import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Iterable
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. "
    "Return JSON only. "
    "Do not include markdown or explanation outside JSON."
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def maybe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(values))


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def looks_like_done(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    patterns = [
        "已完成",
        "完成",
        "成功",
        "任务已完成",
        "验证通过",
        "completed",
        "done",
        "success",
        "finished",
    ]
    return any(token in lowered for token in patterns)


def parse_markdown_links(text: str):
    return re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", text or "")


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def add_judge_args(parser):
    parser.add_argument("--judge-api-base", default=os.environ.get("JUDGE_API_BASE", ""))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--judge-timeout", type=float, default=float(os.environ.get("JUDGE_TIMEOUT", "90.0")))
    return parser


def get_judge_config(args):
    base_url = getattr(args, "judge_api_base", "") or os.environ.get("JUDGE_API_BASE", "")
    api_key = getattr(args, "judge_api_key", "") or os.environ.get("JUDGE_API_KEY", "")
    model = getattr(args, "judge_model", "") or os.environ.get("JUDGE_MODEL", "")
    timeout = float(
        getattr(args, "judge_timeout", 90.0) or os.environ.get("JUDGE_TIMEOUT", "90.0")
    )
    return base_url, api_key, model, timeout


def judge_enabled(args) -> bool:
    base_url, _, model, _ = get_judge_config(args)
    return bool(base_url and model)


def call_judge(base_url: str, api_key: str, model: str, user_prompt: str, timeout: float = 90.0, max_retries: int = 3):
    """调用 LLM judge，支持重试机制"""
    import time

    last_error = None
    for attempt in range(max_retries):
        try:
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key or "dummy",
                model=model,
                temperature=0,
                timeout=timeout,
            )
            response = llm.invoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(
                    str(block.get("text", "")) if isinstance(block, dict) else str(block)
                    for block in content
                )
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            print(f"[WARN] Judge JSON decode failed (attempt {attempt + 1}/{max_retries}): {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Judge API call failed (attempt {attempt + 1}/{max_retries}): {last_error}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 指数退避

    # 所有重试都失败了，返回默认结构
    print(f"[ERROR] Judge failed after {max_retries} attempts: {last_error}")
    return {"error": last_error, "verdict": "UNKNOWN"}
