#!/usr/bin/env python3
"""
实验3评估脚本: 任务尝试次数与成功率关系分析

分析在不同尝试次数下任务成功率的演变。

使用方式:
    python experiments/evaluate_experiment_03.py \
        --scores output/training_free_grpo/trajectory_scores.jsonl \
        --out-dir experiments/results/exp03
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置样式
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 11


def load_existing_results(out_dir: Path) -> tuple[list, set]:
    """加载已存在的结果，返回 (results列表, 已完成prompt_id集合)"""
    details_file = out_dir / "details.jsonl"
    if not details_file.exists():
        return [], set()

    completed_ids = set()
    existing_results = []
    with details_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                existing_results.append(r)
                completed_ids.add(r.get("prompt_id"))
            except Exception:
                continue
    print(f"Loaded {len(existing_results)} existing results from {details_file}")
    return existing_results, completed_ids


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


def generate_charts(results: list, summary: dict, out_dir: Path):
    """生成图表"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 累计成功率折线图
    fig, ax = plt.subplots(figsize=(10, 6))

    attempt_nums = [1, 2, 3, 4, 5]
    success_rates = []
    for n in attempt_nums:
        key = f"success_rate_after_{n}_attempts"
        if key in summary:
            success_rates.append(summary[key] * 100)
        else:
            success_rates.append(0)

    ax.plot(
        attempt_nums,
        success_rates,
        "o-",
        linewidth=2.5,
        markersize=10,
        color="#3498db",
        label="Success Rate",
    )

    # 添加数据标签
    for x, y in zip(attempt_nums, success_rates):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Number of Attempts", fontsize=12)
    ax.set_ylabel("Cumulative Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate vs. Number of Attempts", fontsize=14, fontweight="bold")
    ax.set_xticks(attempt_nums)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")

    plt.tight_layout()
    plt.savefig(fig_dir / "success_rate_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. 尝试次数分布柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    attempt_counts = defaultdict(int)
    for r in results:
        attempt_counts[r["total_attempts"]] += 1

    max_attempts = max(attempt_counts.keys()) if attempt_counts else 1
    x_vals = list(range(1, max_attempts + 1))
    y_vals = [attempt_counts.get(x, 0) for x in x_vals]

    bars = ax.bar(x_vals, y_vals, color="#2ecc71", alpha=0.8, edgecolor="#27ae60")

    for bar, count in zip(bars, y_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.1,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Number of Attempts", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Distribution of Attempt Counts", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_dir / "attempt_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. 边际收益图
    fig, ax = plt.subplots(figsize=(10, 6))

    marginal_improvements = [0]  # 第一次尝试没有边际收益
    for n in range(2, 6):
        curr_key = f"success_rate_after_{n}_attempts"
        prev_key = f"success_rate_after_{n-1}_attempts"
        if curr_key in summary and prev_key in summary:
            improvement = (summary[curr_key] - summary[prev_key]) * 100
            marginal_improvements.append(improvement)
        else:
            marginal_improvements.append(0)

    colors = [
        "#95a5a6" if x <= 0 else "#e74c3c" if x > 5 else "#2ecc71"
        for x in marginal_improvements
    ]
    bars = ax.bar(
        attempt_nums, marginal_improvements, color=colors, alpha=0.8, edgecolor="black"
    )

    for bar, val in zip(bars, marginal_improvements):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{val:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Number of Attempts", fontsize=12)
    ax.set_ylabel("Marginal Improvement (%)", fontsize=12)
    ax.set_title("Marginal Success Rate Improvement", fontsize=14, fontweight="bold")
    ax.set_xticks(attempt_nums)

    plt.tight_layout()
    plt.savefig(fig_dir / "marginal_improvement.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. 任务完成度概览
    fig, ax = plt.subplots(figsize=(8, 6))

    total_tasks = summary.get("total_prompts", 0)
    avg_attempts = summary.get("avg_attempts_per_task", 0)

    # 计算一次性成功率
    first_try_success = 0
    for r in results:
        if r["total_attempts"] >= 1 and r["cumulative"][0]["success"]:
            first_try_success += 1
    first_try_rate = first_try_success / total_tasks * 100 if total_tasks > 0 else 0

    metrics = ["Total Tasks", "Avg Attempts", "1st Try Success Rate (%)"]
    values = [total_tasks, avg_attempts, first_try_rate]

    ax.axis("off")
    table_data = [
        [m, f"{v:.1f}" if isinstance(v, float) else v] for m, v in zip(metrics, values)
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Experiment Overview", fontsize=14, fontweight="bold", y=0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Charts saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="分析任务尝试次数与成功率关系")
    parser.add_argument(
        "--scores", required=True, help="collect.py 生成的 trajectory_scores.jsonl"
    )
    parser.add_argument("--out-dir", required=True, help="输出目录")
    args = parser.parse_args()

    # 加载数据（trajectory_scores.jsonl，直接包含 score）
    scores_path = Path(args.scores)
    scores = load_trajectories(scores_path)
    print(f"Loaded {len(scores)} score records")

    # 按prompt_id分组
    by_prompt = defaultdict(list)
    for record in scores:
        pid = record.get("prompt_id")
        if pid is not None:
            by_prompt[pid].append(record)

    print(f"Found {len(by_prompt)} unique prompts")

    # 支持断点续跑
    existing_results, completed_ids = load_existing_results(Path(args.out_dir))

    # 统计每个prompt的尝试次数
    prompt_stats = []
    for pid, records in by_prompt.items():
        if pid in completed_ids:
            continue
        sorted_records = sorted(records, key=lambda x: x.get("attempt_id", 0))

        attempts = []
        for r in sorted_records:
            score_data = r.get("score", {})
            overall_score = (
                score_data.get("overall_score")
                if isinstance(score_data, dict)
                else None
            )
            # overall_score 范围 0-10，>= 5 视为成功
            success = overall_score is not None and overall_score >= 5.0
            attempts.append(
                {
                    "attempt_id": r.get("attempt_id"),
                    "overall_score": overall_score,
                    "success": success,
                }
            )

        prompt_stats.append(
            {
                "prompt_id": pid,
                "prompt": records[0].get("prompt", ""),
                "total_attempts": len(attempts),
                "attempts": attempts,
            }
        )

    # 计算累计成功率
    results = []
    for ps in prompt_stats:
        success_at_or_before = []
        for i, att in enumerate(ps["attempts"]):
            attempts_so_far = ps["attempts"][: i + 1]
            success_count = sum(1 for a in attempts_so_far if a["success"] is True)
            success_at_or_before.append(
                {
                    "attempt_id": att["attempt_id"],
                    "success": att["success"],
                    "cumulative_success_rate": success_count / (i + 1),
                }
            )

        results.append(
            {
                "prompt_id": ps["prompt_id"],
                "prompt": ps["prompt"],
                "total_attempts": ps["total_attempts"],
                "cumulative": success_at_or_before,
            }
        )

    # 汇总统计
    summary = {
        "total_prompts": len(results),
        "avg_attempts_per_task": (
            mean([r["total_attempts"] for r in results]) if results else 0
        ),
    }

    for n in [1, 2, 3, 4, 5]:
        tasks_with_n_attempts = [r for r in results if r["total_attempts"] >= n]
        if tasks_with_n_attempts:
            success_count = sum(
                1
                for r in tasks_with_n_attempts
                if r["cumulative"][n - 1]["cumulative_success_rate"] > 0
            )
            summary[f"success_rate_after_{n}_attempts"] = success_count / len(
                tasks_with_n_attempts
            )
            summary[f"tasks_with_{n}_attempts"] = len(tasks_with_n_attempts)

    best_n = 1
    best_improvement = 0
    for n in [2, 3, 4, 5]:
        key = f"success_rate_after_{n}_attempts"
        prev_key = f"success_rate_after_{n-1}_attempts"
        if key in summary and prev_key in summary:
            improvement = summary[key] - summary[prev_key]
            if improvement > best_improvement:
                best_improvement = improvement
                best_n = n

    summary["recommended_max_attempts"] = best_n
    summary["improvement_at_best_n"] = best_improvement

    # 保存结果（追加新结果）
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 追加新结果到 details.jsonl
    with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总统计（基于所有结果：existing + new）
    all_results = existing_results + results
    summary["total_prompts"] = len(all_results)
    summary["avg_attempts_per_task"] = (
        mean([r["total_attempts"] for r in all_results]) if all_results else 0
    )

    for n in [1, 2, 3, 4, 5]:
        tasks_with_n_attempts = [r for r in all_results if r["total_attempts"] >= n]
        if tasks_with_n_attempts:
            success_count = sum(
                1
                for r in tasks_with_n_attempts
                if r["cumulative"][n - 1]["cumulative_success_rate"] > 0
            )
            summary[f"success_rate_after_{n}_attempts"] = success_count / len(
                tasks_with_n_attempts
            )
            summary[f"tasks_with_{n}_attempts"] = len(tasks_with_n_attempts)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 生成图表
    generate_charts(all_results, summary, out_dir)


if __name__ == "__main__":
    main()
