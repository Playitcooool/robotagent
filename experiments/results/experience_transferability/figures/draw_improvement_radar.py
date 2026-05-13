#!/usr/bin/env python3
"""
生成经验强化效果雷达图：按难度分组展示 with_exp 相对 without_exp 的得分提升百分比
"""

import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DATA_FILE = Path(__file__).parent.parent / "details.jsonl"
OUT_FILE = Path(__file__).parent / "improvement_radar_by_difficulty.png"


def load_data():
    """加载 details.jsonl，按难度和模式分组计算平均提升"""
    by_difficulty = defaultdict(lambda: {"with_exp": [], "without_exp": []})

    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line.strip())
            diff = r["difficulty"]
            mode = r["mode"]
            score = r.get("score", {}).get("overall_score", 0)
            by_difficulty[diff][mode].append(score)

    # 计算每个难度下 with_exp 相对 without_exp 的平均提升百分比
    # 提升 = (with_exp平均分 - without_exp平均分) / without_exp平均分 * 100
    improvements = {}
    for diff, scores in by_difficulty.items():
        with_exp_avg = mean(scores["with_exp"]) if scores["with_exp"] else 0
        without_exp_avg = mean(scores["without_exp"]) if scores["without_exp"] else 0
        if without_exp_avg > 0:
            pct_improvement = (with_exp_avg - without_exp_avg) / without_exp_avg * 100
        else:
            pct_improvement = 0
        improvements[diff] = {
            "with_exp_avg": with_exp_avg,
            "without_exp_avg": without_exp_avg,
            "improvement_pct": pct_improvement,
            "count": len(scores["with_exp"]),
        }
    return improvements


def draw_radar(improvements):
    """绘制雷达图"""
    difficulties = ["easy", "medium", "hard"]
    diff_labels = {"easy": "简单", "medium": "中等", "hard": "困难"}

    # 角度
    N = len(difficulties)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 数值
    values = [improvements.get(d, {}).get("improvement_pct", 0) for d in difficulties]
    values += values[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 绘制填充区域
    ax.fill(angles, values, color="#42a5f5", alpha=0.25)
    ax.plot(angles, values, "o-", color="#1565c0", linewidth=2.5, markersize=10)

    # 标注数值（放在节点外侧）
    for angle, val, diff in zip(angles[:-1], values[:-1], difficulties):
        ax.text(angle, val + 10, f"{val:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#1565c0")

    # 轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([diff_labels[d] for d in difficulties], fontsize=13, fontweight="bold")

    # 网格
    ax.yaxis.grid(True, color="#90a4ae", linestyle="--", linewidth=0.8)
    ax.xaxis.grid(True, color="#90a4ae", linestyle="--", linewidth=0.8)

    # 0 刻度线
    ax.plot(angles, [0] * len(angles), color="#ef5350", linewidth=1.5, linestyle="--")

    # 图例信息
    info_parts = []
    for d in difficulties:
        info = improvements.get(d, {})
        cnt = info.get("count", 0)
        with_avg = info.get("with_exp_avg", 0)
        without_avg = info.get("without_exp_avg", 0)
        imp = info.get("improvement_pct", 0)
        info_parts.append(f"{diff_labels[d]}（n={cnt}）: {with_avg:.1f} vs {without_avg:.1f} → +{imp:.1f}%")

    legend_text = "\n".join(info_parts)
    fig.text(0.5, 0.02, legend_text, ha="center", fontsize=9, color="#616161",
             style="italic", bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

    fig.tight_layout()
    out = OUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    improvements = load_data()
    print("各难度提升详情：")
    for d, v in improvements.items():
        print(f"  {d}: with_exp={v['with_exp_avg']:.2f}, without_exp={v['without_exp_avg']:.2f}, "
              f"提升={v['improvement_pct']:.1f}%, n={v['count']}")
    draw_radar(improvements)