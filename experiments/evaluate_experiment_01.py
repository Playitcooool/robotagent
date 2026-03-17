#!/usr/bin/env python3
"""
实验1评估脚本: 机器人仿真任务执行质量评估

使用外部LLM Judge评估仿真任务的执行质量。

使用方式:
    python evaluate_experiment_01.py \
        --trajectories ../trajectories.jsonl \
        --out-dir results/exp01

图表输出到 results/exp01/figures/
"""

import argparse
import json
import os
import re
import yaml
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============ 配置 ============

JUDGE_SYSTEM_PROMPT = """你是一个严格的机器人任务评估专家。
评估每个仿真任务的执行质量。
只返回JSON，不要包含markdown或解释。"""


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


def call_judge(llm: ChatOpenAI, prompt: str, max_retries: int = 3):
    """调用LLM Judge"""
    import time

    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke([
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(
                    str(block.get("text", "")) if isinstance(block, dict) else str(block)
                    for block in content
                )
            # 提取JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            print(f"[WARN] Judge JSON decode failed (attempt {attempt + 1}/{max_retries}): {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Judge API call failed (attempt {attempt + 1}/{max_retries}): {last_error}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    print(f"[ERROR] Judge failed after {max_retries} attempts: {last_error}")
    return {"error": last_error, "verdict": "UNKNOWN"}


def build_evaluation_prompt(traj: dict) -> str:
    """构建评估prompt"""
    prompt = traj.get("prompt", "")
    response = traj.get("response", "")

    messages = traj.get("messages", [])
    tool_results = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name") == "task":
            tool_results.append(msg.get("content", ""))

    tool_summary = "\n".join(tool_results[-3:]) if tool_results else "无工具调用记录"

    return f"""评估以下机器人仿真任务的执行质量。

任务描述: {prompt}

执行结果: {response[-500:] if response else '无响应'}

工具返回: {tool_summary[-500:] if tool_summary else '无'}

请评估并返回JSON:
```json
{{
    "task_completion": 1-5,
    "position_accuracy": 1-5,
    "trajectory_quality": 1-5,
    "overall_score": 1-5,
    "verdict": "SUCCESS 或 FAIL"
}}
```"""


def generate_charts(results: list, out_dir: Path):
    """生成图表"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 提取评分数据
    scores = ["task_completion", "position_accuracy", "trajectory_quality", "overall_score"]
    score_data = {s: [] for s in scores}
    verdicts = []

    for r in results:
        j = r.get("judgment", {})
        for s in scores:
            v = j.get(s)
            if v:
                score_data[s].append(v)
        verdict = j.get("verdict", "UNKNOWN")
        verdicts.append(verdict)

    # 1. 成功/失败饼图
    fig, ax = plt.subplots(figsize=(8, 6))
    success_count = sum(1 for v in verdicts if v == "SUCCESS")
    fail_count = sum(1 for v in verdicts if v == "FAIL")
    unknown_count = len(verdicts) - success_count - fail_count

    labels = ['Success', 'Fail', 'Unknown']
    sizes = [success_count, fail_count, unknown_count]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0)

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90
        )
        ax.set_title('Task Success Rate', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / "success_rate_pie.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 2. 评分柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    score_means = [mean(score_data[s]) if score_data[s] else 0 for s in scores]
    score_stds = []
    for s in scores:
        if len(score_data[s]) > 1:
            import statistics
            score_stds.append(statistics.stdev(score_data[s]) if len(score_data[s]) > 1 else 0)
        else:
            score_stds.append(0)

    x_labels = ['Task\nCompletion', 'Position\nAccuracy', 'Trajectory\nQuality', 'Overall\nScore']
    bars = ax.bar(x_labels, score_means, yerr=score_stds, capsize=5,
                  color=['#3498db', '#9b59b6', '#1abc9c', '#e67e22'], alpha=0.8)

    ax.set_ylim(0, 5.5)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title('Evaluation Scores by Dimension', fontsize=14, fontweight='bold')
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Baseline (3.0)')

    # 添加数值标签
    for bar, mean_val in zip(bars, score_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(fig_dir / "score_bars.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 评分分布箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    score_lists = [score_data[s] for s in scores]
    bp = ax.boxplot(score_lists, labels=x_labels, patch_artist=True)

    colors_box = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylim(0, 6)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Charts saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="评估机器人仿真任务质量")
    parser.add_argument("--trajectories", required=True, help="轨迹数据JSONL文件")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--judge-api-base", default=None, help="Judge API Base URL")
    parser.add_argument("--judge-api-key", default=None, help="Judge API Key")
    parser.add_argument("--judge-model", default=None, help="Judge模型名称")
    parser.add_argument("--limit", type=int, default=0, help="限制评估数量")
    parser.add_argument("--skip-judge", action="store_true", help="跳过Judge，仅生成图表")
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)
    judge_config = config.get("judge", {})

    api_base = args.judge_api_base or os.environ.get("JUDGE_API_BASE") or judge_config.get("api_base", "")
    api_key = args.judge_api_key or os.environ.get("JUDGE_API_KEY") or judge_config.get("api_key", "")
    model = args.judge_model or os.environ.get("JUDGE_MODEL") or judge_config.get("model", "deepseek-chat")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 如果有已存在的结果，加载它们用于生成图表
    details_path = out_dir / "details.jsonl"
    if details_path.exists():
        results = []
        with details_path.open("r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
        print(f"Loaded {len(results)} existing results")
        generate_charts(results, out_dir)
        return

    if args.skip_judge:
        print("No existing results found. Run without --skip-judge first.")
        return

    if not api_base or not model:
        raise ValueError("需要配置 judge API（在 config.yaml 中）")

    # 加载数据
    traj_path = Path(args.trajectories)
    trajectories = load_trajectories(traj_path)
    if args.limit > 0:
        trajectories = trajectories[:args.limit]

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Using Judge: {model} @ {api_base}")

    # 初始化LLM
    llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key or "dummy",
        model=model,
        temperature=0,
        timeout=judge_config.get("timeout", 60),
    )

    # 评估
    results = []
    for i, traj in enumerate(trajectories):
        print(f"[{i+1}/{len(trajectories)}] Evaluating prompt_id={traj.get('prompt_id')}, attempt={traj.get('attempt_id')}")
        prompt = build_evaluation_prompt(traj)
        judgment = call_judge(llm, prompt)

        result = {
            "prompt_id": traj.get("prompt_id"),
            "attempt_id": traj.get("attempt_id"),
            "prompt": traj.get("prompt"),
            "judgment": judgment
        }
        results.append(result)

        with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 汇总统计
    summary = {
        "total": len(results),
        "success_count": sum(1 for r in results if r["judgment"].get("verdict") == "SUCCESS"),
        "fail_count": sum(1 for r in results if r["judgment"].get("verdict") == "FAIL"),
    }

    scores = ["task_completion", "position_accuracy", "trajectory_quality", "overall_score"]
    for score in scores:
        values = [r["judgment"].get(score, 0) for r in results if r["judgment"].get(score)]
        if values:
            summary[f"avg_{score}"] = mean(values)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 生成图表
    generate_charts(results, out_dir)


if __name__ == "__main__":
    main()
