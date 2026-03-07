import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Iterable
from urllib.parse import urlparse

import requests


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


def get_judge_config(args):
    base_url = getattr(args, "base_url", "") or os.environ.get("JUDGE_BASE_URL", "")
    api_key = getattr(args, "api_key", "") or os.environ.get("JUDGE_API_KEY", "")
    model = getattr(args, "model", "") or os.environ.get("JUDGE_MODEL", "")
    return base_url, api_key, model


def judge_enabled(args) -> bool:
    base_url, _, model = get_judge_config(args)
    return bool(base_url and model)


def call_judge(base_url: str, api_key: str, model: str, user_prompt: str, timeout: float = 90.0):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

