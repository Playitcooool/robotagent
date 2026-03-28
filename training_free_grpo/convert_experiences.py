"""
将 collect.py 产生的 external_memory.json 转换为 agent_experiences.json 格式。

用法:
    python convert_experiences.py --input output/training_free_grpo/external_memory.json \\
                                  --output prompts/agent_experiences.json \\
                                  --min-score 6.0 \\
                                  --max-items 50
"""

import argparse
import json
from pathlib import Path


def normalize_text(text: str) -> str:
    """去除空格和标点，用于相似度比较。"""
    return text.strip()


def is_duplicate(new_item: str, existing: list[str], threshold: float = 0.75) -> bool:
    """
    简单字符串相似度去重。
    如果 new_item 和已有条目有 >threshold 的字符重叠，认为重复。
    """
    new_lower = new_item.lower()
    new_words = set(new_lower.split())
    if not new_words:
        return False
    for ex in existing:
        ex_lower = ex.lower()
        ex_words = set(ex_lower.split())
        overlap = len(new_words & ex_words)
        # Jaccard-like similarity
        union = len(new_words | ex_words)
        if union > 0 and overlap / union >= threshold:
            return True
    return False


def dedupe(items: list[str], threshold: float = 0.75) -> list[str]:
    """对 items 做去重，保持顺序。"""
    result = []
    for item in items:
        norm = normalize_text(item)
        if not norm:
            continue
        if not is_duplicate(item, result, threshold):
            result.append(item)
    return result


def extract_keywords_from_summaries(experiences: list[dict]) -> set[str]:
    """从 summary 中提取高频关键词。"""
    keyword_count: dict[str, int] = {}
    keywords_blacklist = {
        "机器人", "仿真", "任务", "系统", "操作", "执行", "工具",
        "调用", "问题", "解决", "结果", "阶段", "具体", "信息",
    }
    for exp in experiences:
        words = exp.get("summary", "").split()
        for w in words:
            w = w.strip("，。、：：,.[]()（）")
            if len(w) >= 3 and w not in keywords_blacklist:
                keyword_count[w] = keyword_count.get(w, 0) + 1
    # Return top keywords
    return {k for k, v in sorted(keyword_count.items(), key=lambda x: -x[1])[:20]}


def build_items(experiences: list[dict], max_items: int = 50) -> list[str]:
    """从经验中提取并去重行为准则。"""
    raw_items: list[tuple[str, float]] = []  # (item_text, score)

    for exp in experiences:
        score = exp.get("score", 0)
        for p in exp.get("principles", []):
            raw_items.append((p, score))
        for d in exp.get("dos", []):
            raw_items.append((d, score))
        for d in exp.get("donts", []):
            raw_items.append((d, score))

    # Sort by score descending, then dedupe
    raw_items.sort(key=lambda x: -x[1])
    items = dedupe([item for item, _ in raw_items])
    return items[:max_items]


def build_high_score_points(experiences: list[dict], threshold: float = 7.5) -> str:
    """从高分经验的 summary 提炼核心要点。"""
    high = [e for e in experiences if e.get("score", 0) >= threshold]
    if not high:
        return ""
    points: list[str] = []
    for exp in high:
        summary = exp.get("summary", "").strip()
        if summary:
            points.append(summary)
    # Deduplicate points
    unique_points = dedupe(points, threshold=0.6)
    lines = ["高分轨迹特点："]
    for pt in unique_points[:5]:
        lines.append(f"- {pt}")
    return "\n".join(lines)


def get_agent_types(exp: dict) -> list[str]:
    """从经验中获取 agent_type 列表。"""
    types = exp.get("agent_type", [])
    if isinstance(types, list) and types:
        return types
    if isinstance(types, str) and types:
        return [types]
    return ["main"]


def group_by_agent_type(experiences: list[dict]) -> dict[str, list[dict]]:
    """按 agent_type 分组，同一经验可能属于多个 agent_type。"""
    groups: dict[str, list[dict]] = {}
    for exp in experiences:
        for atype in get_agent_types(exp):
            groups.setdefault(atype, []).append(exp)
    return groups


def build_header(agent_type: str, experiences: list[dict]) -> str:
    """生成 header 标题。"""
    keywords = extract_keywords_from_summaries(experiences)
    high_kw = ", ".join(sorted(keywords)[:5])
    return f"## 实际轨迹中的常见错误与高分要点"


def convert(
    input_path: str,
    output_path: str,
    min_score: float = 6.0,
    max_items: int = 50,
    high_score_threshold: float = 7.5,
) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    all_experiences = data.get("experiences", [])
    print(f"[convert] loaded {len(all_experiences)} experiences from {input_path}")

    # Filter by score
    filtered = [e for e in all_experiences if e.get("score", 0) >= min_score]
    print(f"[convert] after min-score {min_score}: {len(filtered)} experiences")

    # Group by agent_type
    groups = group_by_agent_type(filtered)

    result: dict[str, dict] = {}
    for atype, exps in groups.items():
        key = f"{atype}_agent"
        items = build_items(exps, max_items=max_items)
        header = build_header(atype, exps)
        high_points = build_high_score_points(exps, threshold=high_score_threshold)
        result[key] = {
            "header": header,
            "items": items,
            "高分要点": high_points,
            "agent_type": [atype],
        }
        print(f"[convert] {key}: {len(exps)} experiences -> {len(items)} items")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[convert] wrote {len(result)} agent types to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert collect experiences to agent_experiences.json")
    parser.add_argument("--input", default="output/training_free_grpo/external_memory.json")
    parser.add_argument("--output", default="prompts/agent_experiences.json")
    parser.add_argument("--min-score", type=float, default=6.0)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--high-score-threshold", type=float, default=7.5)
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        min_score=args.min_score,
        max_items=args.max_items,
        high_score_threshold=args.high_score_threshold,
    )
