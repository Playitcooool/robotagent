#!/usr/bin/env python3
"""
实验4评估脚本: 经验迁移性分析 (Experience Transferability Analysis)

研究问题: 从训练轨迹中提炼的经验（agent_experiences.json），能否泛化到全新的测试任务？

实验设计:
- 测试集: experiments/data/test_queries.jsonl (20个新提示，不在训练轨迹中)
  - easy (6个): 简单单步操作
  - medium (6个): 多步骤或参数调节
  - hard (8个): 精密控制/协同规划/边界条件
- 对比: 每个提示分别以"有经验注入"和"无经验注入"两种方式运行
  - 有经验: backend 注入了 prompts/agent_experiences.json 的经验段落
  - 无经验: backend 使用 clean base prompt
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
import time as time_module
from pathlib import Path
from statistics import mean, stdev

import yaml
from langchain_openai import ChatOpenAI


def load_test_queries(path: Path) -> list:
    """加载测试查询"""
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


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


def call_judge(llm: ChatOpenAI, prompt: str, max_retries: int = 3) -> dict:
    """调用LLM Judge评分"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke([
                {"role": "system", "content": "你是一个严格的质量评审专家。只返回JSON格式的评分结果。"},
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
            time_module.sleep(2 ** attempt)

    print(f"[ERROR] Judge failed after {max_retries} attempts: {last_error}")
    return {"overall_score": 0.0, "task_completion": 0.0, "correctness": 0.0,
            "clarity": 0.0, "robustness": 0.0, "conciseness": 0.0,
            "brief_reason": last_error or "judge failed"}


def build_judge_prompt(query: dict, response: str, experience_context: str = "") -> str:
    """构建Judge评估prompt"""
    return f"""请对以下机器人任务响应进行严格评分。

**测试提示**: {query['prompt']}
**难度**: {query.get('difficulty', 'unknown')}
**经验上下文**:
{experience_context}

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


def load_existing_results(out_dir: Path) -> tuple[list, set]:
    """加载已存在的结果，支持断点续跑"""
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
                key = (r.get("query_id"), r.get("mode"))  # mode: with_exp / without_exp
                completed_keys.add(key)
            except Exception:
                continue
    print(f"Loaded {len(existing_results)} existing results from {details_file}")
    return existing_results, completed_keys


async def run_query_with_agent(query: dict, with_experience: bool,
                                 agent_config: dict) -> tuple[str, str]:
    """
    通过后端API运行查询

    Returns:
        tuple of (response_text, error_message)
    """
    import requests

    backend_url = agent_config.get("backend_url", "http://localhost:8000")
    session_id = f"exp04_{query['id']}_{'exp' if with_experience else 'noexp'}"

    headers = {"Content-Type": "application/json"}
    # 适配 backend 的 ChatIn schema: message 对应 prompt, 用 exp_mode 控制带/不带经验
    payload = {
        "message": query["prompt"],
        "session_id": session_id,
        "enabled_tools": [],
    }

    try:
        # 使用流式API收集完整响应，通过 exp_mode 选择带/不带经验的 agent
        response_text = ""
        async with requests.post(
            f"{backend_url}/api/chat/send?exp_mode={'without_exp' if not with_experience else 'with_exp'}",
            json=payload,
            headers=headers,
            timeout=agent_config.get("timeout", 120),
            stream=True,
        ) as resp:
            if not resp.ok:
                return "", f"HTTP {resp.status_code}"

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                if data.get("type") == "error":
                    return "", data.get("error", "unknown error")
                if data.get("type") == "delta":
                    response_text += data.get("text", "")
                if data.get("type") == "done":
                    break

        return response_text.strip(), ""
    except requests.Timeout:
        return "", "timeout"
    except Exception as e:
        return "", str(e)


def generate_report(results: list, out_dir: Path):
    """生成分析报告和图表"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 按 query_id 分组
    by_query = {}
    for r in results:
        qid = r["query_id"]
        if qid not in by_query:
            by_query[qid] = {}
        by_query[qid][r["mode"]] = r

    # 按难度分组
    diff_scores = {"easy": {"with_exp": [], "without_exp": []},
                   "medium": {"with_exp": [], "without_exp": []},
                   "hard": {"with_exp": [], "without_exp": []}}
    improvements = []

    for qid, modes in by_query.items():
        w_exp = modes.get("with_exp", {}).get("score", {})
        wo_exp = modes.get("without_exp", {}).get("score", {})

        if w_exp and wo_exp:
            delta = w_exp.get("overall_score", 0) - wo_exp.get("overall_score", 0)
            improvements.append(delta)

            difficulty = modes.get("with_exp", {}).get("difficulty", "unknown")
            if difficulty in diff_scores:
                diff_scores[difficulty]["with_exp"].append(w_exp.get("overall_score", 0))
                diff_scores[difficulty]["without_exp"].append(wo_exp.get("overall_score", 0))

    # 生成图表
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. 经验提升率分布
        if improvements:
            axes[0].hist(improvements, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
            axes[0].axvline(mean(improvements), color='red', linestyle='--', linewidth=2,
                             label=f'Mean: {mean(improvements):.2f}')
            axes[0].set_xlabel('Score Improvement (with - without)', fontsize=12)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title('Experience Benefit Distribution', fontsize=14, fontweight='bold')
            axes[0].legend()

        # 2. 按难度分组对比
        difficulties = ['easy', 'medium', 'hard']
        with_scores = [mean(diff_scores[d]["with_exp"]) if diff_scores[d]["with_exp"] else 0
                       for d in difficulties]
        without_scores = [mean(diff_scores[d]["without_exp"]) if diff_scores[d]["without_exp"] else 0
                          for d in difficulties]

        x = np.arange(len(difficulties))
        width = 0.35
        axes[1].bar(x - width/2, with_scores, width, label='With Experience', color='#2ecc71', alpha=0.8)
        axes[1].bar(x + width/2, without_scores, width, label='Without Experience', color='#e74c3c', alpha=0.8)
        axes[1].set_ylabel('Average Score', fontsize=12)
        axes[1].set_title('Performance by Task Difficulty', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(difficulties)
        axes[1].legend()
        axes[1].set_ylim(0, 10)

        plt.tight_layout()
        plt.savefig(fig_dir / "exp04_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 3. 每个 query 的对比条形图
        if by_query:
            fig, ax = plt.subplots(figsize=(14, 6))
            sorted_qids = sorted(by_query.keys())
            x = np.arange(len(sorted_qids))
            width = 0.35

            w_scores = [by_query[q].get("with_exp", {}).get("score", {}).get("overall_score", 0)
                        for q in sorted_qids]
            wo_scores = [by_query[q].get("without_exp", {}).get("score", {}).get("overall_score", 0)
                         for q in sorted_qids]

            ax.bar(x - width/2, w_scores, width, label='With Experience', color='#2ecc71', alpha=0.8)
            ax.bar(x + width/2, wo_scores, width, label='Without Experience', color='#e74c3c', alpha=0.8)

            # 标注难度
            diff_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
            for i, qid in enumerate(sorted_qids):
                diff = by_query[q].get("with_exp", {}).get("difficulty", "unknown")
                color = diff_colors.get(diff, '#95a5a6')
                ax.axvline(i, color=color, linestyle=':', alpha=0.6, linewidth=2)

            ax.set_xlabel('Query ID', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Per-Query: With vs Without Experience (green=easy, orange=medium, red=hard)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_qids)
            ax.legend()
            ax.set_ylim(0, 10)

            plt.tight_layout()
            plt.savefig(fig_dir / "per_query_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Charts saved to {fig_dir}/")

    except ImportError:
        print("[WARN] matplotlib not available, skipping charts")


async def main():
    parser = argparse.ArgumentParser(description="实验4: 经验迁移性分析")
    parser.add_argument("--test-queries", required=True, help="测试查询JSONL文件")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制测试数量")
    parser.add_argument("--experiences-only", action="store_true",
                        help="只运行有经验注入的测试（跳过无经验对照）")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    judge_config = config.get("judge", {})
    agent_config = config.get("agent", {})

    api_base = os.environ.get("JUDGE_API_BASE") or judge_config.get("api_base", "")
    api_key = os.environ.get("JUDGE_API_KEY") or judge_config.get("api_key", "")
    model = os.environ.get("JUDGE_MODEL") or judge_config.get("model", "deepseek-chat")

    if not api_base or not model:
        raise ValueError("需要配置 judge API（在 config.yaml 或环境变量中）")

    # 加载测试数据和经验
    queries = load_test_queries(Path(args.test_queries))
    if args.limit > 0:
        queries = queries[:args.limit]

    print(f"Loaded {len(queries)} test queries")

    # 初始化Judge LLM
    llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key or "dummy",
        model=model,
        temperature=0,
        timeout=judge_config.get("timeout", 120),
    )

    # 输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 支持断点续跑
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
                print(f"[{run_idx}/{total_runs}] Skipping query_id={query['id']}, mode={mode} (already completed)")
                continue

            print(f"[{run_idx}/{total_runs}] Query {query['id']} ({query['category']}), mode={mode}")

            # 运行查询
            response, error = await run_query_with_agent(
                query,
                with_experience=(mode == "with_exp"),
                agent_config=agent_config,
            )

            if error:
                print(f"  [ERROR] {error}")
                score_data = {"overall_score": 0.0, "brief_reason": f"运行错误: {error}"}
            else:
                # 调用Judge评分
                judge_prompt = build_judge_prompt(query, response, "")
                score_data = call_judge(llm, judge_prompt)
                print(f"  Score: {score_data.get('overall_score', 'N/A'):.1f}/10 - {score_data.get('brief_reason', '')[:50]}")

            result = {
                "query_id": query["id"],
                "difficulty": query["difficulty"],
                "prompt": query["prompt"],
                "mode": mode,
                "response": response[:500] if response else "",
                "error": error,
                "score": score_data,
            }
            results.append(result)

            # 保存中间结果
            with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # 每个query跑完一次with_exp后等待一下，避免并发过高
        await asyncio.sleep(1)

    # 汇总分析
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
        }
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Experiment 4 Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 生成图表
    generate_report(results, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
