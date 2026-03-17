#!/usr/bin/env python3
"""
Experience 提取与精炼脚本

从 trajectory_scores.jsonl 中提取高质量的 experiences，
并生成模块化的系统提示词片段，用于添加到 agent 的提示词中。

使用方式:
    python extract_experiences.py \
        --input ../output/training_free_grpo/trajectory_scores.jsonl \
        --output experiences/ \
        --top-k 10
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean


def load_trajectories(path: Path):
    """加载轨迹数据"""
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


def extract_experiences(trajectories: list) -> dict:
    """从轨迹中提取 experiences"""
    experiences = {
        "high_quality": [],   # 高分案例的优点
        "low_quality": [],    # 低分案例的缺点
        "successful_patterns": [],  # 成功的模式
        "failed_patterns": [],      # 失败的模式
    }

    for traj in trajectories:
        score = traj.get("score", {})
        overall = score.get("overall_score", 0)
        prompt = traj.get("prompt", "")
        response = traj.get("response", "")[:300]

        # 提取优缺点
        pros = score.get("pros", [])
        cons = score.get("cons", [])
        brief = score.get("brief_reason", "")

        entry = {
            "prompt": prompt,
            "response": response,
            "score": overall,
            "pros": pros,
            "cons": cons,
            "brief_reason": brief,
        }

        if overall >= 6.0:
            experiences["high_quality"].append(entry)
            if pros:
                experiences["successful_patterns"].extend(pros)
        elif overall <= 4.0:
            experiences["low_quality"].append(entry)
            if cons:
                experiences["failed_patterns"].extend(cons)

    # 去重
    experiences["successful_patterns"] = list(set(experiences["successful_patterns"]))
    experiences["failed_patterns"] = list(set(experiences["failed_patterns"]))

    return experiences


def generate_system_prompt(experiences: dict, top_k: int = 10) -> str:
    """生成模块化的系统提示词"""

    # 高质量案例
    high_quality = experiences.get("high_quality", [])[:top_k]
    successful_patterns = experiences.get("successful_patterns", [])[:top_k]
    failed_patterns = experiences.get("failed_patterns", [])[:top_k]

    prompt = """# 机器人仿真任务执行经验

## 最佳实践 (Best Practices)
以下是从成功案例中提炼的最佳实践：
"""

    for i, pattern in enumerate(successful_patterns, 1):
        prompt += f"\n{i}. {pattern}"

    prompt += """

## 常见错误与避免方法
以下是从失败案例中总结的常见错误：
"""

    for i, pattern in enumerate(failed_patterns, 1):
        prompt += f"\n{i}. {pattern}"

    prompt += """

## 高质量执行示例
以下是得分较高的执行示例（带评分和简要说明）：
"""

    for i, entry in enumerate(high_quality, 1):
        prompt += f"""
### 示例 {i} (得分: {entry['score']}/10)
**任务**: {entry['prompt'][:100]}...
**执行要点**: {entry.get('brief_reason', '')[:200]}
"""

    return prompt


def generate_json_experiences(experiences: dict, top_k: int = 10) -> dict:
    """生成 JSON 格式的 experiences（用于程序化加载）"""

    high_quality = experiences.get("high_quality", [])[:top_k]
    successful_patterns = experiences.get("successful_patterns", [])[:top_k]
    failed_patterns = experiences.get("failed_patterns", [])[:top_k]

    return {
        "successful_patterns": successful_patterns,
        "failed_patterns": failed_patterns,
        "high_quality_examples": [
            {
                "prompt": e["prompt"][:200],
                "score": e["score"],
                "brief_reason": e.get("brief_reason", "")[:300]
            }
            for e in high_quality
        ],
        "statistics": {
            "total_high_quality": len(experiences.get("high_quality", [])),
            "total_low_quality": len(experiences.get("low_quality", [])),
            "unique_successful_patterns": len(experiences.get("successful_patterns", [])),
            "unique_failed_patterns": len(experiences.get("failed_patterns", [])),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="提取并精炼 experiences")
    parser.add_argument("--input", required=True, help="轨迹数据JSONL文件")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--top-k", type=int, default=10, help="提取的top-k数量")
    args = parser.parse_args()

    # 加载数据
    input_path = Path(args.input)
    trajectories = load_trajectories(input_path)
    print(f"Loaded {len(trajectories)} trajectories")

    # 提取 experiences
    experiences = extract_experiences(trajectories)
    print(f"High quality: {len(experiences['high_quality'])}")
    print(f"Low quality: {len(experiences['low_quality'])}")
    print(f"Unique successful patterns: {len(experiences['successful_patterns'])}")
    print(f"Unique failed patterns: {len(experiences['failed_patterns'])}")

    # 生成输出
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 系统提示词格式
    system_prompt = generate_system_prompt(experiences, args.top_k)
    (output_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
    print(f"Saved system_prompt.md")

    # 2. JSON 格式
    json_experiences = generate_json_experiences(experiences, args.top_k)
    (output_dir / "experiences.json").write_text(
        json.dumps(json_experiences, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved experiences.json")

    # 3. 分类整理
    categories = defaultdict(list)
    for traj in trajectories:
        prompt = traj.get("prompt", "")
        score = traj.get("score", {}).get("overall_score", 0)

        # 简单的任务分类
        if "抓取" in prompt or "grab" in prompt.lower():
            categories["抓取任务"].append(traj)
        elif "推" in prompt or "push" in prompt.lower():
            categories["推动任务"].append(traj)
        elif "放置" in prompt or "place" in prompt.lower():
            categories["放置任务"].append(traj)
        elif "初始化" in prompt or "init" in prompt.lower():
            categories["初始化任务"].append(traj)
        else:
            categories["其他任务"].append(traj)

    category_summary = {}
    for cat, trajs in categories.items():
        scores = [t.get("score", {}).get("overall_score", 0) for t in trajs if t.get("score")]
        category_summary[cat] = {
            "count": len(trajs),
            "avg_score": mean(scores) if scores else 0,
        }

    (output_dir / "category_summary.json").write_text(
        json.dumps(category_summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved category_summary.json")

    print(f"\n=== Summary ===")
    print(f"Output directory: {output_dir}")
    print(f"Total trajectories: {len(trajectories)}")
    for cat, stats in category_summary.items():
        print(f"  {cat}: {stats['count']} tasks, avg score: {stats['avg_score']:.2f}")


if __name__ == "__main__":
    main()
