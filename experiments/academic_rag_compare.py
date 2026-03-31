#!/usr/bin/env python3
"""
学术搜索 RAG 对比脚本：比较 baseline（无搜索）vs Agentic RAG（arXiv + Tavily 搜索）的评估结果。

Usage:
    python academic_rag_compare.py
"""

import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_DIR = RESULTS_DIR / "academic_baseline"
AGENT_DIR = RESULTS_DIR / "academic_rag_agent"


def load_summary(d: Path) -> dict:
    with (d / "summary.json").open() as f:
        return json.load(f)


def load_details(d: Path) -> list:
    rows = []
    with (d / "details.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def match_by_query_id(baseline_details: list, agent_details: list) -> list:
    """按 query_id 匹配两条结果，返回配对列表"""
    agent_map = {r["query_id"]: r for r in agent_details}
    pairs = []
    for b in baseline_details:
        qid = b.get("query_id")
        if qid in agent_map:
            pairs.append((b, agent_map[qid]))
    return pairs


def compute_improvement(pairs: list) -> dict:
    """计算每个维度的提升率（agent - baseline）"""
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    improvements = {d: [] for d in dims}
    for b, a in pairs:
        bs = b.get("score", {})
        as_ = a.get("score", {})
        for d in dims:
            b_val = bs.get(d, 0)
            a_val = as_.get(d, 0)
            improvements[d].append(a_val - b_val)
    return {d: mean(v) for d, v in improvements.items()}


def plot_summary_comparison(baseline_sum: dict, agent_sum: dict, out_dir: Path):
    """绘制综合维度对比柱状图"""
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    labels = [
        "Relevance",
        "Accuracy",
        "Completeness",
        "Citation",
        "Overall",
    ]

    b_vals = [baseline_sum[f"avg_{d}_pct"] for d in dims]
    a_vals = [agent_sum[f"avg_{d}_pct"] for d in dims]

    x = range(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width / 2 for i in x], b_vals, width, label="Baseline (no search)", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], a_vals, width, label="Agentic RAG (arXiv+Tavily)", color="#3498db", alpha=0.8)

    ax.set_ylabel("Score (%)")
    ax.set_title("Exp1: Baseline vs Agentic RAG — Average Scores by Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    for bar, val in zip(bars1, b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, a_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80% target")
    ax.axhline(y=60, color="orange", linestyle="--", alpha=0.5, label="60% baseline")

    plt.tight_layout()
    out_path = out_dir / "comparison_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_improvement_distribution(pairs: list, out_dir: Path):
    """绘制每个 query 的提升分布（直方图）"""
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, dim in zip(axes, dims):
        diffs = []
        for b, a in pairs:
            bs = b.get("score", {})
            as_ = a.get("score", {})
            diffs.append(as_.get(dim, 0) - bs.get(dim, 0))

        color = "#2ecc71" if mean(diffs) > 0 else "#e74c3c"
        ax.hist(diffs, bins=10, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=mean(diffs), color="red", linestyle="-", linewidth=2, label=f"μ={mean(diffs):.2f}")
        ax.set_title(dim.replace("_", "\n").title())
        ax.legend(fontsize=7)
        ax.set_xlabel("Δ score")

    plt.suptitle("Score Improvement Distribution (Agent − Baseline) per Query", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "comparison_improvement_dist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_win_rate(pairs: list, out_dir: Path):
    """统计 Agent 胜出/持平/落败的 query 比例"""
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]

    results = {d: {"win": 0, "tie": 0, "lose": 0} for d in dims}
    for b, a in pairs:
        bs = b.get("score", {})
        as_ = a.get("score", {})
        for d in dims:
            delta = as_.get(d, 0) - bs.get(d, 0)
            if delta > 0:
                results[d]["win"] += 1
            elif delta == 0:
                results[d]["tie"] += 1
            else:
                results[d]["lose"] += 1

    labels = ["Relevance", "Accuracy", "Completeness", "Citation", "Overall"]
    n = len(pairs)
    win_rates = [results[d]["win"] / n * 100 for d in dims]
    tie_rates = [results[d]["tie"] / n * 100 for d in dims]
    lose_rates = [results[d]["lose"] / n * 100 for d in dims]

    x = range(len(dims))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, win_rates, label="Agent wins", color="#2ecc71")
    ax.bar(x, tie_rates, bottom=win_rates, label="Tie", color="#f39c12")
    ax.bar(x, lose_rates, bottom=[w + t for w, t in zip(win_rates, tie_rates)],
           label="Baseline wins", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Win/Tie/Lose Rate per Dimension (n={n} queries)")
    ax.legend()

    for i, (w, t, l) in enumerate(zip(win_rates, tie_rates, lose_rates)):
        ax.text(i, w / 2, f"{w:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "comparison_win_rate.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    baseline_sum = load_summary(BASELINE_DIR)
    agent_sum = load_summary(AGENT_DIR)

    baseline_details = load_details(BASELINE_DIR)
    agent_details = load_details(AGENT_DIR)

    pairs = match_by_query_id(baseline_details, agent_details)
    print(f"Matched {len(pairs)} queries between baseline and agent")
    print(f"Baseline: {baseline_sum['total']} queries, Agent: {agent_sum['total']} queries")

    out_dir = RESULTS_DIR / "academic_rag_comparison"
    out_dir.mkdir(exist_ok=True)

    # ========== Print summary table ==========
    print("\n" + "=" * 60)
    print("SUMMARY: Baseline vs Agentic RAG")
    print("=" * 60)
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    header = f"{'Dimension':<18} {'Baseline':>12} {'Agent RAG':>12} {'Gap':>10} {'Δ':>8}"
    print(header)
    print("-" * 60)
    for d in dims:
        b_pct = baseline_sum[f"avg_{d}_pct"]
        a_pct = agent_sum[f"avg_{d}_pct"]
        gap = a_pct - b_pct
        marker = "★" if gap > 0 else ""
        print(f"{d.capitalize():<18} {b_pct:>10.1f}% {a_pct:>10.1f}% {gap:>+9.1f}% {marker}")

    improvements = compute_improvement(pairs)
    print("\n" + "-" * 60)
    print(f"Overall improvement: {improvements['overall_score']:+.3f} (avg per-query delta)")

    # ========== Generate charts ==========
    print("\nGenerating charts...")
    plot_summary_comparison(baseline_sum, agent_sum, out_dir)
    plot_improvement_distribution(pairs, out_dir)
    plot_win_rate(pairs, out_dir)

    # ========== Save comparison JSON ==========
    comparison = {
        "baseline_summary": baseline_sum,
        "agent_summary": agent_sum,
        "matched_queries": len(pairs),
        "improvements_per_query": improvements,
        "dimensions": {d: {
            "baseline_pct": baseline_sum[f"avg_{d}_pct"],
            "agent_pct": agent_sum[f"avg_{d}_pct"],
            "gap": agent_sum[f"avg_{d}_pct"] - baseline_sum[f"avg_{d}_pct"],
            "avg_delta": improvements[d],
        } for d in dims},
    }
    with (out_dir / "comparison.json").open("w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
