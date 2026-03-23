#!/usr/bin/env python3
"""
实验4评估脚本: 经验迁移性分析 (Experience Transferability Analysis)

研究问题: 从训练轨迹中提炼的经验，能否泛化到全新的测试任务？

实验设计:
- 测试集: experiments/data/test_queries.jsonl (20个新提示，不在训练轨迹中)
  - easy (6个): 简单单步操作
  - medium (6个): 多步骤或参数调节
  - hard (8个): 精密控制/协同规划/边界条件
- 对比: 每个提示分别以"有经验注入"和"无经验注入"两种方式运行
- Agent: 本地部署模型（与 collect.py 相同）
- 评估: 使用 DeepSeek Judge 独立评分两者
- 分析: 计算经验带来的提升率，按难度分组分析泛化边界

使用方式:
    python evaluate_experiment_04.py \
        --test-queries experiments/data/test_queries.jsonl \
        --out-dir results/exp04
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time as time_module
from pathlib import Path
from statistics import mean, stdev
from typing import Any

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

import yaml
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import ToolRetryMiddleware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(root_dir) / "config" / "config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Agent builder (mirrors collect.py build_agent)
# ---------------------------------------------------------------------------

async def build_exp_agent(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    with_experiences: bool,
    request_timeout_s: float | None = 600.0,
) -> Any:
    """Build an agent. If with_experiences=True, inject from agent_experiences.json."""
    from deepagents import create_deep_agent
    import tools.GeneralTool as GeneralToolModule
    from tools.SubAgentTool import init_subagents, _load_agent_experiences, build_experience_suffix

    general_tools = []
    for func_name in GeneralToolModule.__all__:
        general_tools.append(getattr(GeneralToolModule, func_name))

    # Load experiences from JSON
    experiences = _load_agent_experiences() if with_experiences else []

    # Build system prompt with or without experience suffix
    if with_experiences and experiences:
        exp_suffix = build_experience_suffix(experiences)
        full_system_prompt = system_prompt + exp_suffix
    else:
        full_system_prompt = system_prompt

    subagents = list(await init_subagents(experiences=experiences))

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        request_timeout=request_timeout_s,
    )

    return create_deep_agent(
        model=chat,
        tools=general_tools,
        system_prompt=full_system_prompt,
        subagents=subagents,
        middleware=[
            ToolRetryMiddleware(
                max_retries=1,
                backoff_factor=1.0,
                initial_delay=1.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Judge (DeepSeek)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是一个严格的质量评审专家。只返回JSON格式的评分结果。"""


def call_judge(llm: ChatOpenAI, prompt: str, max_retries: int = 3) -> dict:
    """Call LLM Judge to score a response."""
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
                    str(block.get("text", "")) if isinstance(block, dict) else str(block)
                    for block in content
                )
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            time_module.sleep(2 ** attempt)

    return {
        "overall_score": 0.0,
        "task_completion": 0.0,
        "correctness": 0.0,
        "clarity": 0.0,
        "robustness": 0.0,
        "conciseness": 0.0,
        "brief_reason": last_error or "judge failed",
    }


def build_judge_prompt(query: dict, response: str, mode: str) -> str:
    """Build prompt for judge."""
    return f"""请对以下机器人任务响应进行严格评分。

**测试提示**: {query['prompt']}
**难度**: {query.get('difficulty', 'unknown')}
**模式**: {mode}

**响应内容**:
{response[:1500] if response else '（无响应）'}

请返回严格JSON评分（0-10分）：
{{
    "overall_score": 综合分数,
    "task_completion": 任务完成度,
    "correctness": 正确性,
    "clarity": 清晰度,
    "robustness": 鲁棒性,
    "conciseness": 简洁性,
    "brief_reason": 简要评分理由（1-2句话，中文）
}}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_queries(path: Path) -> list:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def load_existing_results(out_dir: Path) -> tuple[list, set]:
    """Load existing results, return (results, completed_keys)."""
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
                key = (r.get("query_id"), r.get("mode"))
                completed_keys.add(key)
            except Exception:
                continue
    print(f"Loaded {len(existing_results)} existing results from {details_file}")
    return existing_results, completed_keys


# ---------------------------------------------------------------------------
# Run agent on a query
# ---------------------------------------------------------------------------

async def run_agent_query(
    agent, query: dict, timeout_s: float = 300.0
) -> tuple[str, str]:
    """Run agent on a single query, return (response_text, error)."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": query["prompt"]}]}),
            timeout=timeout_s,
        )
        # Extract response text from result
        messages = result.get("messages", []) if isinstance(result, dict) else []
        # Get the last assistant message
        response_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break
        return response_text, ""
    except asyncio.TimeoutError:
        return "", "timeout"
    except Exception as e:
        return "", str(e)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_report(results: list, out_dir: Path):
    """Generate charts."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    # Group by query
    by_query = {}
    for r in results:
        qid = r["query_id"]
        if qid not in by_query:
            by_query[qid] = {}
        by_query[qid][r["mode"]] = r

    improvements = []
    diff_improvements = {"easy": [], "medium": [], "hard": []}

    for qid, modes in by_query.items():
        w_exp = modes.get("with_exp", {}).get("score", {})
        wo_exp = modes.get("without_exp", {}).get("score", {})

        if w_exp and wo_exp:
            delta = w_exp.get("overall_score", 0) - wo_exp.get("overall_score", 0)
            improvements.append(delta)

            difficulty = modes.get("with_exp", {}).get("difficulty", "unknown")
            if difficulty in diff_improvements:
                diff_improvements[difficulty].append(delta)

    # 1. Improvement distribution
    if improvements:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(improvements, bins=10, color="#3498db", alpha=0.7, edgecolor="black")
        ax.axvline(mean(improvements), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean(improvements):.2f}")
        ax.set_xlabel("Score Improvement (with - without)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Experience Benefit Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "improvement_hist.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 2. By difficulty
    difficulties = ["easy", "medium", "hard"]
    with_scores = []
    without_scores = []
    for d in difficulties:
        w = diff_improvements[d]
        with_scores.append(mean(w) if w else 0)
        wo_vals = [
            by_query[q].get("without_exp", {}).get("score", {}).get("overall_score", 0)
            for q in by_query
            if by_query[q].get("without_exp", {}).get("difficulty") == d
            and by_query[q].get("without_exp", {}).get("score", {})
        ]
        without_scores.append(mean(wo_vals) if wo_vals else 0)

    x = np.arange(len(difficulties))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, with_scores, width, label="With Experience", color="#2ecc71", alpha=0.8)
    ax.bar(x + width / 2, without_scores, width, label="Without Experience", color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Performance by Task Difficulty", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.savefig(fig_dir / "by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Per-query comparison
    if by_query:
        fig, ax = plt.subplots(figsize=(14, 6))
        sorted_qids = sorted(by_query.keys())
        x = np.arange(len(sorted_qids))

        w_scores = [by_query[q].get("with_exp", {}).get("score", {}).get("overall_score", 0) for q in sorted_qids]
        wo_scores = [by_query[q].get("without_exp", {}).get("score", {}).get("overall_score", 0) for q in sorted_qids]

        ax.bar(x - width / 2, w_scores, width, label="With Experience", color="#2ecc71", alpha=0.8)
        ax.bar(x + width / 2, wo_scores, width, label="Without Experience", color="#e74c3c", alpha=0.8)

        diff_colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
        for i, qid in enumerate(sorted_qids):
            diff = by_query[qid].get("with_exp", {}).get("difficulty", "unknown")
            color = diff_colors.get(diff, "#95a5a6")
            ax.axvline(i, color=color, linestyle=":", alpha=0.6, linewidth=2)

        ax.set_xlabel("Query ID", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Query: With vs Without Experience (green=easy, orange=medium, red=hard)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_qids)
        ax.legend()
        ax.set_ylim(0, 10)
        plt.tight_layout()
        plt.savefig(fig_dir / "per_query_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Charts saved to {fig_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="实验4: 经验迁移性分析（直接Agent对比）")
    parser.add_argument("--test-queries", required=True, help="测试查询JSONL文件")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制测试数量")
    parser.add_argument("--experiences-only", action="store_true", help="只运行有经验版本")
    args = parser.parse_args()

    config = load_config()

    # Agent config (local deployment)
    agent_base_url = os.environ.get("AGENT_MODEL_URL") or config.get("model_url", "http://localhost:1234/v1")
    agent_model = os.environ.get("AGENT_MODEL") or config.get("llm", "")
    agent_api_key = os.environ.get("AGENT_API_KEY") or config.get("api_key", "no_need")

    # Judge config (DeepSeek)
    judge_cfg = config.get("judge", {})
    judge_api_base = os.environ.get("JUDGE_API_BASE") or judge_cfg.get("api_base", "")
    judge_api_key = os.environ.get("JUDGE_API_KEY") or judge_cfg.get("api_key", "")
    judge_model = os.environ.get("JUDGE_MODEL") or judge_cfg.get("model", "deepseek-chat")

    if not judge_api_base or not judge_model:
        raise ValueError("需要配置 judge API（在 config.yaml 或环境变量中）")

    # Load test queries
    queries = load_test_queries(Path(args.test_queries))
    if args.limit > 0:
        queries = queries[:args.limit]
    print(f"Loaded {len(queries)} test queries")

    # Init judge LLM
    judge_llm = ChatOpenAI(
        base_url=judge_api_base,
        api_key=judge_api_key or "dummy",
        model=judge_model,
        temperature=0,
        timeout=judge_cfg.get("timeout", 120),
    )

    # Build two agents at startup (rebuilt each query to ensure clean state)
    from prompts import MainAgentPrompt

    print(f"Building agents with local model: {agent_model} @ {agent_base_url}")

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    existing_results, completed_keys = load_existing_results(out_dir)
    results = existing_results

    modes_to_run = ["with_exp"]
    if not args.experiences_only:
        modes_to_run.append("without_exp")

    total_runs = len(queries) * len(modes_to_run)
    run_idx = 0

    for query in queries:
        for mode in modes_to_run:
            run_idx += 1
            key = (query["id"], mode)
            if key in completed_keys:
                print(f"[{run_idx}/{total_runs}] Skipping query_id={query['id']}, mode={mode} (already done)")
                continue

            print(f"[{run_idx}/{total_runs}] Query {query['id']} ({query.get('difficulty', '?')}), mode={mode}")

            # Build agent for this run
            with_exp = (mode == "with_exp")
            agent = await build_exp_agent(
                base_url=agent_base_url,
                model=agent_model,
                api_key=agent_api_key,
                system_prompt=MainAgentPrompt.SYSTEM_PROMPT,
                with_experiences=with_exp,
                request_timeout_s=600.0,
            )

            # Run query
            response, error = await run_agent_query(agent, query, timeout_s=300.0)

            if error:
                print(f"  [ERROR] {error}")
                score_data = {"overall_score": 0.0, "brief_reason": f"运行错误: {error}"}
            else:
                # Judge scoring
                judge_prompt = build_judge_prompt(query, response, mode)
                score_data = call_judge(judge_llm, judge_prompt)
                print(f"  Score: {score_data.get('overall_score', 'N/A'):.1f}/10 - {score_data.get('brief_reason', '')[:60]}")

            result = {
                "query_id": query["id"],
                "difficulty": query.get("difficulty", "unknown"),
                "prompt": query["prompt"],
                "mode": mode,
                "response": response[:500] if response else "",
                "error": error,
                "score": score_data,
            }
            results.append(result)

            # Save
            with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Pause between queries to avoid overwhelming
        await asyncio.sleep(1)

    # Summary
    by_query = {}
    for r in results:
        qid = r["query_id"]
        if qid not in by_query:
            by_query[qid] = {}
        by_query[qid][r["mode"]] = r

    improvements = []
    diff_improvements = {"easy": [], "medium": [], "hard": []}

    for qid, modes in by_query.items():
        w_exp = modes.get("with_exp", {}).get("score", {})
        wo_exp = modes.get("without_exp", {}).get("score", {})

        if w_exp and wo_exp:
            delta = w_exp.get("overall_score", 0) - wo_exp.get("overall_score", 0)
            improvements.append(delta)

            difficulty = modes.get("with_exp", {}).get("difficulty", "unknown")
            if difficulty in diff_improvements:
                diff_improvements[difficulty].append(delta)

    diff_stats = {}
    for d, vals in diff_improvements.items():
        diff_stats[d] = {
            "mean": round(mean(vals), 3) if vals else 0,
            "count": len(vals),
        }

    summary = {
        "total_queries": len(queries),
        "total_results": len(results),
        "difficulty_counts": {
            "easy": sum(1 for q in queries if q.get("difficulty") == "easy"),
            "medium": sum(1 for q in queries if q.get("difficulty") == "medium"),
            "hard": sum(1 for q in queries if q.get("difficulty") == "hard"),
        },
        "improvements": {
            "mean": round(mean(improvements), 3) if improvements else 0,
            "std": round(stdev(improvements), 3) if len(improvements) > 1 else 0,
            "positive_ratio": round(sum(1 for i in improvements if i > 0) / len(improvements), 3) if improvements else 0,
            "by_difficulty": diff_stats,
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Experiment 4 Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    generate_report(results, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
