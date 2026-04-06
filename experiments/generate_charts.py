#!/usr/bin/env python3
"""
生成所有实验图表（中文标签）
直接读取 experiments/results/ 下的实验数据，无需重新运行实验。
"""

import json
import sys
from pathlib import Path
from statistics import mean
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#f8f9fa"
plt.rcParams["grid.color"] = "#d0d0d0"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

RESULTS_DIR = Path(__file__).parent / "results"


# ──────────────────────────────────────────────
# 实验一：学术搜索质量评估（Agentic RAG vs Baseline）
# ──────────────────────────────────────────────

def load_summary(d):
    with (d / "summary.json").open() as f:
        return json.load(f)


def plot_exp1_agent_scores():
    """实验一：Agentic RAG 各评估维度得分柱状图"""
    agent_dir = RESULTS_DIR / "academic_rag_agent"
    baseline_dir = RESULTS_DIR / "academic_baseline"
    agent_sum = load_summary(agent_dir)
    baseline_sum = load_summary(baseline_dir)

    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    # 中文标签映射
    labels = ["相关度\n(Relevance)", "准确性\n(Accuracy)", "完整性\n(Completeness)", "引用质量\n(Citation)", "综合得分\n(Overall)"]
    agent_vals = [agent_sum[f"avg_{d}_pct"] for d in dims]
    baseline_vals = [baseline_sum[f"avg_{d}_pct"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline（无检索）", color="#ef9a9a", edgecolor="#c62828", linewidth=1.2)
    bars2 = ax.bar(x + width/2, agent_vals, width, label="Agentic RAG（arXiv+Tavily）", color="#90caf9", edgecolor="#1565c0", linewidth=1.2)

    ax.set_ylabel("得分（%）", fontsize=13)
    ax.set_title("实验一：学术搜索问答质量评估——各维度得分对比", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=11)

    # 数据标签
    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#c62828", fontweight="bold")
    for bar, val in zip(bars2, agent_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#1565c0", fontweight="bold")

    # 参考线
    ax.axhline(y=80, color="#4caf50", linestyle="--", alpha=0.7, linewidth=1.5, label="80% 目标线")
    ax.axhline(y=60, color="#ff9800", linestyle="--", alpha=0.7, linewidth=1.5, label="60% 基线")

    fig.tight_layout()
    out = RESULTS_DIR / "academic_rag_agent" / "figures" / "score_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # 得分分布直方图
    details_path = agent_dir / "details.jsonl"
    if details_path.exists():
        overall_scores = []
        with details_path.open() as f:
            for line in f:
                r = json.loads(line.strip())
                s = r.get("score", {}).get("overall_score", 0)
                if s > 0:
                    overall_scores.append(s)

        fig, ax = plt.subplots(figsize=(9, 5))
        counts, bins, patches = ax.hist(overall_scores, bins=5, edgecolor="white", color="#42a5f5", alpha=0.9)
        # 高分颜色深，低分颜色浅
        for i, (cnt, p) in enumerate(zip(counts, patches)):
            p.set_facecolor(plt.cm.Blues(0.3 + 0.6 * (i / 4)))
        ax.set_xlabel("综合得分（1-5分）", fontsize=12)
        ax.set_ylabel("频次", fontsize=12)
        ax.set_title("Agentic RAG 综合得分分布", fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(range(1, 6))
        for cnt, bin_left in zip(counts, bins[:-1]):
            ax.text(bin_left + 0.2, cnt + 0.3, f"{int(cnt)}", ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout()
        out2 = RESULTS_DIR / "academic_rag_agent" / "figures" / "score_distribution.png"
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out2}")


def plot_exp1_comparison():
    """实验一：Baseline vs Agentic RAG 综合对比"""
    baseline_sum = load_summary(RESULTS_DIR / "academic_baseline")
    agent_sum = load_summary(RESULTS_DIR / "academic_rag_agent")

    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    labels = ["相关度", "准确性", "完整性", "引用质量", "综合得分"]
    baseline_vals = [baseline_sum[f"avg_{d}_pct"] for d in dims]
    agent_vals = [agent_sum[f"avg_{d}_pct"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline（无检索）", color="#ef9a9a", edgecolor="#c62828")
    bars2 = ax.bar(x + width/2, agent_vals, width, label="Agentic RAG（arXiv+Tavily）", color="#90caf9", edgecolor="#1565c0")

    # 提升箭头标注
    for i, (b, a) in enumerate(zip(baseline_vals, agent_vals)):
        gap = a - b
        if gap > 0:
            ax.annotate("", xy=(i + width/2 + 0.15, a + 2), xytext=(i + width/2 + 0.15, b + 2),
                       arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5))
            ax.text(i + width/2 + 0.2, (a + b) / 2 + 2, f"+{gap:.1f}%", fontsize=9, color="#2e7d32", fontweight="bold")

    ax.set_ylabel("得分（%）", fontsize=13)
    ax.set_title("Agentic RAG vs Baseline 各维度得分对比", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=11)

    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#c62828")
    for bar, val in zip(bars2, agent_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#1565c0")

    fig.tight_layout()
    out = RESULTS_DIR / "academic_rag_comparison" / "figures" / "comparison_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# 实验二：任务尝试次数与成功率关系（Pass@k）
# ──────────────────────────────────────────────

def plot_exp2_pass_at_k():
    """实验二：累计成功率曲线（Pass@k）"""
    summary = load_summary(RESULTS_DIR / "task_attempt_analysis")

    attempt_nums = [1, 2, 3, 4]
    success_rates = [
        summary.get(f"success_rate_after_{n}_attempts", 0) * 100
        for n in attempt_nums
    ]
    sample_sizes = [
        summary.get(f"tasks_with_{n}_attempts", 0)
        for n in attempt_nums
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(attempt_nums, success_rates, "o-", linewidth=2.8, markersize=12,
            color="#1976d2", markerfacecolor="#bbdefb", markeredgewidth=2, markeredgecolor="#1565c0",
            label="累计成功率", zorder=3)

    # 填充曲线下方
    ax.fill_between(attempt_nums, success_rates, alpha=0.1, color="#1976d2")

    # 数据标签 + 样本量
    for x, y, n in zip(attempt_nums, success_rates, sample_sizes):
        ax.annotate(f"{y:.1f}%\n(n={n})", (x, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=11, fontweight="bold", color="#1565c0")

    ax.set_xlabel("尝试次数 (k)", fontsize=13)
    ax.set_ylabel("累计成功率（%）", fontsize=13)
    ax.set_title("实验二：Agent 多次尝试累计成功率（Pass@k）", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(attempt_nums)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, 4.5)
    ax.grid(axis="y", alpha=0.4)

    # 50% 基线
    ax.axhline(y=50, color="#9e9e9e", linestyle="--", linewidth=1.5, alpha=0.7, label="50% 基线")
    ax.legend(fontsize=11)

    fig.tight_layout()
    out = RESULTS_DIR / "task_attempt_analysis" / "figures" / "success_rate_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_exp2_marginal_improvement():
    """实验二：边际收益柱状图"""
    summary = load_summary(RESULTS_DIR / "task_attempt_analysis")

    attempt_nums = [1, 2, 3, 4]
    marginal = [0.0]  # 第一次尝试无边际收益
    for n in range(2, 5):
        curr = summary.get(f"success_rate_after_{n}_attempts", 0)
        prev = summary.get(f"success_rate_after_{n-1}_attempts", 0)
        marginal.append((curr - prev) * 100)

    colors = ["#90a4ae"] + [
        "#e53935" if m > 10 else "#43a047" if m > 0 else "#e53935"
        for m in marginal[1:]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(attempt_nums, marginal, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, val in zip(bars, marginal):
        offset = 0.3 if val >= 0 else -1.0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f"{val:+.1f}%", ha="center", fontsize=11, fontweight="bold",
                color="#e53935" if val < 5 else "#2e7d32")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("尝试次数", fontsize=13)
    ax.set_ylabel("边际收益（%）", fontsize=13)
    ax.set_title("实验二：每次尝试对成功率的边际提升", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(attempt_nums)
    ax.set_ylim(min(marginal) - 5, max(marginal) + 8)

    # 说明文字
    ax.text(0.02, 0.97,
            "注：边际收益 = S(k) - S(k-1)\nS(k) 为 k 次尝试内至少成功一次的累计成功率",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            color="#616161", style="italic")

    fig.tight_layout()
    out = RESULTS_DIR / "task_attempt_analysis" / "figures" / "marginal_improvement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_exp2_overview():
    """实验二：概览信息表"""
    summary = load_summary(RESULTS_DIR / "task_attempt_analysis")

    total = summary.get("total_prompts", 0)
    avg_attempts = summary.get("avg_attempts_per_task", 0)
    pass1 = summary.get("success_rate_after_1_attempts", 0) * 100
    pass4 = summary.get("success_rate_after_4_attempts", 0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    cell_data = [
        ["总任务数", f"{total}"],
        ["平均尝试次数", f"{avg_attempts:.2f}"],
        ["Pass@1 成功率", f"{pass1:.1f}%"],
        ["Pass@4 累计成功率", f"{pass4:.1f}%"],
        ["推荐最大尝试次数", "3~4 次"],
    ]

    table = ax.table(
        cellText=cell_data,
        colLabels=["指标", "数值"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.4, 2.2)

    # 表头
    for j in range(2):
        table[(0, j)].set_facecolor("#1565c0")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # 奇偶行颜色
    for i in range(1, len(cell_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor("#e3f2fd" if i % 2 == 1 else "white")

    ax.set_title("实验二：任务尝试次数实验概览", fontsize=15, fontweight="bold", y=0.88)

    fig.tight_layout()
    out = RESULTS_DIR / "task_attempt_analysis" / "figures" / "overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# 实验三：经验回放迁移性分析
# ──────────────────────────────────────────────

def plot_exp3_improvement_hist():
    """实验三：经验提升分布直方图"""
    summary = load_summary(RESULTS_DIR / "experience_transferability")

    mean_imp = summary.get("improvements", {}).get("mean", 0)
    std_imp = summary.get("improvements", {}).get("std", 0)
    positive_ratio = summary.get("improvements", {}).get("positive_ratio", 0) * 100
    diff_stats = summary.get("improvements", {}).get("by_difficulty", {})

    fig, ax = plt.subplots(figsize=(8, 5))
    difficulties = ["easy", "medium", "hard"]
    diff_labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    colors = {"easy": "#66bb6a", "medium": "#ffa726", "hard": "#ef5350"}
    means = [diff_stats.get(d, {}).get("mean", 0) for d in difficulties]
    counts = [diff_stats.get(d, {}).get("count", 0) for d in difficulties]

    x = np.arange(len(difficulties))
    bars = ax.bar(x, means, color=[colors[d] for d in difficulties], edgecolor="white", linewidth=1.5, width=0.55)

    for bar, m, c in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"μ={m:.2f}\n(n={c})", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([diff_labels[d] for d in difficulties], fontsize=12)
    ax.set_ylabel("平均得分提升（with - without）", fontsize=12)
    ax.set_title("实验三：经验回放效果按任务难度分组分析", fontsize=14, fontweight="bold", pad=12)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylim(min(means) - 1, max(means) + 2 if max(means) > 0 else 2)

    # 图例
    legend_patches = [mpatches.Patch(color=colors[d], label=diff_labels[d]) for d in difficulties]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper right")

    fig.tight_layout()
    out = RESULTS_DIR / "experience_transferability" / "figures" / "improvement_hist.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_exp3_summary():
    """实验三：汇总信息"""
    summary = load_summary(RESULTS_DIR / "experience_transferability")
    mean_imp = summary.get("improvements", {}).get("mean", 0)
    std_imp = summary.get("improvements", {}).get("std", 0)
    positive_ratio = summary.get("improvements", {}).get("positive_ratio", 0) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")

    cell_data = [
        ["平均得分提升", f"{mean_imp:.3f}"],
        ["提升标准差", f"{std_imp:.3f}"],
        ["正向提升比例", f"{positive_ratio:.1f}%"],
        ["总测试任务", f"{summary.get('total_queries', 0)}"],
    ]

    table = ax.table(cellText=cell_data, colLabels=["指标", "数值"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.0)

    for j in range(2):
        table[(0, j)].set_facecolor("#1565c0")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor("#e3f2fd" if i % 2 == 1 else "white")

    ax.set_title("实验三：经验迁移性分析汇总", fontsize=14, fontweight="bold", y=0.88)
    fig.tight_layout()
    out = RESULTS_DIR / "experience_transferability" / "figures" / "overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────
# 综合对比图：所有实验结果汇总
# ──────────────────────────────────────────────

def plot_all_summary():
    """生成一页综合概览图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("多智能体协同机器人仿真系统——实验结果总览", fontsize=16, fontweight="bold", y=1.01)

    # ── 左上：学术搜索得分 ──
    ax = axes[0, 0]
    agent_sum = load_summary(RESULTS_DIR / "academic_rag_agent")
    baseline_sum = load_summary(RESULTS_DIR / "academic_baseline")
    dims = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    labels = ["相关度", "准确性", "完整性", "引用", "综合"]
    b_vals = [baseline_sum[f"avg_{d}_pct"] for d in dims]
    a_vals = [agent_sum[f"avg_{d}_pct"] for d in dims]
    x = np.arange(len(dims))
    ax.bar(x - 0.2, b_vals, 0.35, label="Baseline", color="#ef9a9a", edgecolor="#c62828")
    ax.bar(x + 0.2, a_vals, 0.35, label="Agentic RAG", color="#90caf9", edgecolor="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("(a) Agentic RAG vs Baseline", fontsize=12, fontweight="bold")
    ax.set_ylabel("得分（%）", fontsize=12)
    ax.legend(fontsize=9)

    # ── 右上：Pass@k 曲线 ──
    ax = axes[0, 1]
    summary2 = load_summary(RESULTS_DIR / "task_attempt_analysis")
    ks = [1, 2, 3, 4]
    rates = [summary2.get(f"success_rate_after_{k}_attempts", 0) * 100 for k in ks]
    ax.plot(ks, rates, "o-", color="#1976d2", linewidth=2.5, markersize=10, markerfacecolor="#bbdefb", markeredgewidth=2)
    ax.fill_between(ks, rates, alpha=0.1, color="#1976d2")
    for k, r in zip(ks, rates):
        ax.annotate(f"{r:.1f}%", (k, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.set_title("(b) Agent Pass@k 累计成功率", fontsize=12, fontweight="bold")
    ax.set_xlabel("尝试次数 (k)", fontsize=12)
    ax.set_ylabel("累计成功率（%）", fontsize=12)

    # ── 左下：边际收益 ──
    ax = axes[1, 0]
    marginal = [0.0]
    for n in range(2, 5):
        curr = summary2.get(f"success_rate_after_{n}_attempts", 0)
        prev = summary2.get(f"success_rate_after_{n-1}_attempts", 0)
        marginal.append((curr - prev) * 100)
    colors = ["#90a4ae"] + ["#e53935" if m < 5 else "#43a047" for m in marginal[1:]]
    bars = ax.bar(ks, marginal, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, marginal):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.8,
                f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(ks)
    ax.set_title("(c) 每次尝试的边际收益", fontsize=12, fontweight="bold")
    ax.set_xlabel("尝试次数", fontsize=12)
    ax.set_ylabel("边际收益（%）", fontsize=12)

    # ── 右下：经验回放效果 ──
    ax = axes[1, 1]
    summary3 = load_summary(RESULTS_DIR / "experience_transferability")
    diff_stats = summary3.get("improvements", {}).get("by_difficulty", {})
    difficulties = ["easy", "medium", "hard"]
    diff_labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    colors = {"easy": "#66bb6a", "medium": "#ffa726", "hard": "#ef5350"}
    means = [diff_stats.get(d, {}).get("mean", 0) for d in difficulties]
    ax.bar([diff_labels[d] for d in difficulties], means, color=[colors[d] for d in difficulties], edgecolor="white", width=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("(d) 经验回放提升（按难度）", fontsize=12, fontweight="bold")
    ax.set_ylabel("平均得分提升", fontsize=12)

    fig.tight_layout()
    out = RESULTS_DIR / "all_experiments_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("=" * 60)
    print("开始生成实验图表...")
    print("=" * 60)

    plot_exp1_agent_scores()
    plot_exp1_comparison()
    plot_exp2_pass_at_k()
    plot_exp2_marginal_improvement()
    plot_exp2_overview()
    plot_exp3_improvement_hist()
    plot_exp3_summary()
    plot_all_summary()

    print("=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
