#!/usr/bin/env python3
"""
实验2评估脚本: 机器人仿真任务执行质量评估

使用外部LLM Judge (DeepSeek) 评估机器人仿真任务轨迹的执行质量。
分析不同任务类型（抓取、推、放置等）的执行质量差异。

使用方式:
    python evaluate_experiment_02.py \
        --trajectories ../output/training_free_grpo/trajectories.jsonl \
        --out-dir results/exp02
"""

import argparse
import json
import os
import re
import yaml
from pathlib import Path
from statistics import mean

from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def generate_charts(results: list, out_dir: Path):
    """生成图表"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 提取数据
    success_rates = []
    quality_scores = []

    for r in results:
        j = r.get("judgment", {})
        sr = j.get("success_rate")
        qs = j.get("quality_score")
        if sr is not None:
            success_rates.append(sr)
        if qs is not None:
            quality_scores.append(qs)

    # 1. 成功率分布直方图
    if success_rates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(success_rates, bins=10, color="#3498db", alpha=0.7, edgecolor="black")
        ax.axvline(
            mean(success_rates),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean(success_rates):.1f}%",
        )
        ax.set_xlabel("Success Rate (%)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Success Rate Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "success_rate_hist.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. 质量评分柱状图
    if quality_scores:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(quality_scores, bins=5, color="#9b59b6", alpha=0.7, edgecolor="black")
        ax.axvline(
            mean(quality_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean(quality_scores):.2f}",
        )
        ax.set_xlabel("Quality Score (1-5)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Quality Score Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "quality_score_hist.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Charts saved to {fig_dir}/")


def load_existing_results(out_dir: Path) -> tuple[list, set]:
    """加载已存在的结果，返回 (results列表, 已完成(prompt_id, attempt_id)集合)"""
    details_file = out_dir / "details.jsonl"
    if not details_file.exists():
        return [], set()

    completed_keys = set()
    existing_results = []
    with details_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                existing_results.append(r)
                key = (r.get("prompt_id"), r.get("attempt_id"))
                completed_keys.add(key)
            except Exception:
                continue
    print(f"Loaded {len(existing_results)} existing results from {details_file}")
    return existing_results, completed_keys


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
评估每个任务的执行质量，只返回JSON格式。"""


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
            response = llm.invoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(
                    (
                        str(block.get("text", ""))
                        if isinstance(block, dict)
                        else str(block)
                    )
                    for block in content
                )
            # 提取JSON
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            print(
                f"[WARN] Judge JSON decode failed (attempt {attempt + 1}/{max_retries}): {last_error}"
            )
        except Exception as e:
            last_error = str(e)
            print(
                f"[WARN] Judge API call failed (attempt {attempt + 1}/{max_retries}): {last_error}"
            )

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    print(f"[ERROR] Judge failed after {max_retries} attempts: {last_error}")
    return {
        "error": last_error,
        "success_rate": None,
        "quality_score": None,
    }


def build_evaluation_prompt(traj: dict) -> str:
    """构建评估prompt"""
    prompt = traj.get("prompt", "")
    response = traj.get("response", "")

    # 提取工具调用结果
    messages = traj.get("messages", [])
    tool_results = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name") == "task":
            tool_results.append(msg.get("content", ""))

    tool_summary = "\n".join(tool_results[-3:]) if tool_results else "无工具调用记录"

    return f"""评估以下机器人任务的执行质量。

任务描述: {prompt}

执行结果: {response[-800:] if response else '无响应'}

工具返回: {tool_summary[-800:] if tool_summary else '无'}

请评估并返回JSON:
{{
    "success_rate": 0-100,   // 任务成功率估算(%)(0=完全失败, 100=完美完成)
    "quality_score": 1-5,    // 执行质量评分(1=极差, 5=优秀)
}}
```"""


def main():
    parser = argparse.ArgumentParser(description="使用DeepSeek评估任务执行质量")
    parser.add_argument("--trajectories", required=True, help="轨迹数据JSONL文件")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制评估数量")
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)
    judge_config = config.get("judge", {})

    api_base = os.environ.get("JUDGE_API_BASE") or judge_config.get("api_base", "")
    api_key = os.environ.get("JUDGE_API_KEY") or judge_config.get("api_key", "")
    model = os.environ.get("JUDGE_MODEL") or judge_config.get("model", "deepseek-chat")

    if not api_base or not model:
        raise ValueError("需要配置 judge API（在 config.yaml 中）")

    # 加载数据
    traj_path = Path(args.trajectories)
    trajectories = load_trajectories(traj_path)
    if args.limit > 0:
        trajectories = trajectories[: args.limit]

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Using Judge: {model} @ {api_base}")

    # 初始化LLM
    llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key or "dummy",
        model=model,
        temperature=0,
        timeout=judge_config.get("timeout", 120),
    )

    # 评估
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 支持断点续跑
    existing_results, completed_keys = load_existing_results(out_dir)

    results = existing_results
    for i, traj in enumerate(trajectories):
        key = (traj.get("prompt_id"), traj.get("attempt_id"))
        if key in completed_keys:
            print(
                f"[{i+1}/{len(trajectories)}] Skipping prompt_id={key[0]}, attempt={key[1]} (already completed)"
            )
            continue
        print(
            f"[{i+1}/{len(trajectories)}] Evaluating prompt_id={traj.get('prompt_id')}, attempt={traj.get('attempt_id')}"
        )
        prompt = build_evaluation_prompt(traj)
        judgment = call_judge(llm, prompt)

        result = {
            "prompt_id": traj.get("prompt_id"),
            "attempt_id": traj.get("attempt_id"),
            "prompt": traj.get("prompt"),
            "response": traj.get("response", "")[:200],
            "judgment": judgment,
        }
        results.append(result)

        # 保存中间结果
        with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 汇总统计（仅依赖 success_rate，废弃 verdict）
    success_rates = [
        r["judgment"].get("success_rate", 0)
        for r in results
        if r["judgment"].get("success_rate") is not None
    ]
    quality_scores = [
        r["judgment"].get("quality_score", 0)
        for r in results
        if r["judgment"].get("quality_score") is not None
    ]

    summary = {
        "total": len(results),
        "avg_success_rate": mean(success_rates) if success_rates else None,
        "avg_quality_score": mean(quality_scores) if quality_scores else None,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 生成图表
    generate_charts(results, out_dir)


if __name__ == "__main__":
    main()
