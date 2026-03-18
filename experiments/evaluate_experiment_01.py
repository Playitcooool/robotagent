#!/usr/bin/env python3
"""
实验1评估脚本: Agentic 学术搜索问答系统质量评估

评估使用 Agent + Academic Search 回答机器人领域问题的质量。
创建带有 search 工具的 agent，使用外部LLM Judge (DeepSeek) 进行评估。

使用方式:
    python evaluate_experiment_01.py \
        --queries data/rag_queries.jsonl \
        --out-dir results/exp01_academic_agent
"""

import argparse
import json
import os
import re
import sys
import yaml
import asyncio
from pathlib import Path
from statistics import mean
from collections import defaultdict
from langchain.agents.middleware import ToolCallLimitMiddleware

# 添加项目根目录到路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from tools.GeneralTool import search

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = current_dir / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============ 配置 ============

# Agent 系统提示词
AGENT_SYSTEM_PROMPT = """你是一个专业的问答助手，专门回答机器人领域的问题。

你可以通过调用 search 工具搜索信息来回答问题。
该工具同时搜索网页（Tavily）和学术论文（OpenAlex + arXiv），并自动去重合并结果。

请按照以下步骤回答问题：
1. 分析用户问题，确定需要搜索的关键词
2. 调用 search 工具搜索相关信息
3. 根据搜索结果整理并回答用户问题
4. 在回答中引用来源（论文标题、年份、作者或网页标题、URL）

注意：
- 每个问题最多搜索 1~2 次，避免过度搜索
- 搜索 1 次后已有足够信息就直接回答，不要反复搜索
- 只在搜索结果明显不足或与问题不相关时才重新搜索
- 回答要简洁，控制在 300 字以内
- 引用时优先使用搜索结果中最相关的 1~2 条即可"""

JUDGE_SYSTEM_PROMPT = """你是一个严格的学术问答评估专家。
评估 Agent + 学术搜索系统回答机器人领域问题的质量。
只返回JSON，不要包含markdown或解释。

评分维度：
- relevance: 问题相关性 (1-5) - 搜索结果与问题的相关程度
- accuracy: 答案准确性 (1-5) - 论文信息是否准确
- completeness: 答案完整性 (1-5) - 是否提供了足够的信息
- citation: 引用质量 (1-5) - 引用论文的质量和相关性
- overall_score: 综合得分 (1-5)

返回格式：
{"relevance": 4, "accuracy": 3, "completeness": 4, "citation": 3, "overall_score": 3.5,
 "pros": ["优点1", "优点2"], "cons": ["缺点1"], "brief_reason": "简要说明"}"""


def load_config_full() -> dict:
    """加载完整配置"""
    config_path = root_dir / "config" / "config.yml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_queries(path: Path):
    """加载查询数据"""
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


def create_academic_agent(base_url: str, model: str, api_key: str):
    """创建带有 search 工具的 agent"""
    # 创建 LLM
    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=0.8,
        request_timeout=600,
    )

    # 创建 agent，带有 search 工具
    agent = create_deep_agent(
        model=chat,
        tools=[search],
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(tool_name="search", run_limit=3)],
    )

    return agent


async def call_agent(agent, query: str) -> str:
    """调用 Agent 获取回答"""
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

        # 提取最终回答
        for msg in reversed(result.get("messages", [])):
            cls_name = msg.__class__.__name__
            if cls_name == "AIMessage":
                return str(msg.content).strip()
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()

        # 如果没找到，尝试获取所有内容
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return str(last_msg.content).strip()
            if isinstance(last_msg, dict):
                return str(last_msg.get("content", "")).strip()

        return ""
    except Exception as e:
        print(f"[WARN] Agent call failed: {e}")
        import traceback

        traceback.print_exc()
        return ""


def call_judge(llm: ChatOpenAI, query: str, answer: str, max_retries: int = 3):
    """调用LLM Judge评估回答质量"""
    import time

    prompt = f"""请评估以下 Agent + 学术搜索系统回答的质量：

问题：{query}

Agent回答：{answer}

请严格按照评分标准给出评分和反馈。评估时考虑：
1. 搜索结果是否与问题相关
2. 论文信息是否准确（标题、作者、年份等）
3. 是否提供了足够的信息
4. 引用的论文质量和相关性"""

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
                result = json.loads(json_match.group())
                # 确保有所有必要字段
                result.setdefault("relevance", 0)
                result.setdefault("accuracy", 0)
                result.setdefault("completeness", 0)
                result.setdefault("citation", 0)
                result.setdefault("overall_score", 0)
                result.setdefault("pros", [])
                result.setdefault("cons", [])
                result.setdefault("brief_reason", "")
                return result
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
        time.sleep(1)

    print(f"[ERROR] Judge failed after {max_retries} retries: {last_error}")
    return {
        "relevance": 0,
        "accuracy": 0,
        "completeness": 0,
        "citation": 0,
        "overall_score": 0,
        "pros": [],
        "cons": [],
        "brief_reason": f"Judge failed: {last_error}",
    }


async def evaluate_academic_agent(
    queries: list,
    agent,
    llm,
    model_name: str,
    out_dir: Path,
    delay_between_queries: float = 3.0,
):
    """评估 Agent + 学术搜索系统

    Args:
        delay_between_queries: 每次查询之间的延迟（秒），用于避免 API 限流
    """
    import asyncio

    results = []
    total = len(queries)

    print(f"Loaded {total} queries")
    print(f"Using Agent model: {model_name}")
    print(f"Using Judge: {llm.model_name}")
    print(f"Delay between queries: {delay_between_queries}s")

    for i, q in enumerate(queries, 1):
        query_id = q.get("id", f"query_{i}")
        query_text = q.get("query", "")

        print(f"[{i}/{total}] Evaluating {query_id}: {query_text[:30]}...")

        # 调用 Agent
        answer = await call_agent(agent, query_text)

        if not answer:
            print(f"[WARN] No answer for {query_id}, skipping...")
            results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "answer": "",
                    "score": {
                        "relevance": 0,
                        "accuracy": 0,
                        "completeness": 0,
                        "citation": 0,
                        "overall_score": 0,
                    },
                }
            )
            # 即使失败也等待，避免连续失败触发限制
            if i < total:
                print(f"  Waiting {delay_between_queries}s before next query...")
                await asyncio.sleep(delay_between_queries)
            continue

        # 调用 Judge 评估
        score = call_judge(llm, query_text, answer)

        results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "answer": answer,
                "score": score,
            }
        )

        # 每次查询结束后等待，避免触发 API 限流
        if i < total:
            print(f"  Waiting {delay_between_queries}s before next query...")
            await asyncio.sleep(delay_between_queries)

        # 保存进度
        if i % 5 == 0:
            with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 保存最终结果
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def generate_charts(results: list, out_dir: Path):
    """生成可视化图表"""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 提取分数
    scores = [r.get("score", {}) for r in results if r.get("score")]
    if not scores:
        return

    # 1. 评分维度柱状图
    dimensions = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    avg_scores = {d: mean([s.get(d, 0) for s in scores]) for d in dimensions}

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        avg_scores.keys(),
        avg_scores.values(),
        color=["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"],
    )
    ax.set_ylim(0, 5)
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Agent + Academic Search Answer Quality by Dimension")
    for bar, val in zip(bars, avg_scores.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(figures_dir / "score_bars.png", dpi=150)
    plt.close()

    # 2. 得分分布直方图
    overall_scores = [
        s.get("overall_score", 0) for s in scores if s.get("overall_score", 0) > 0
    ]
    if overall_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(overall_scores, bins=5, edgecolor="black", alpha=0.7, color="#3498db")
        ax.set_xlabel("Overall Score")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Overall Scores")
        ax.set_xticks(range(1, 6))
        plt.tight_layout()
        plt.savefig(figures_dir / "score_distribution.png", dpi=150)
        plt.close()

    # 3. 分类别评分热力图
    categories = defaultdict(list)
    for r in results:
        query = r.get("query", "")
        score = r.get("score", {})
        # 简单分类
        cat = "general"
        if "机器人" in query or "操作" in query:
            cat = "robotics"
        elif "学习" in query or "policy" in query.lower() or "RL" in query:
            cat = "learning"
        elif "RAG" in query or "rag" in query.lower():
            cat = "rag"
        elif "仿真" in query or "sim" in query.lower():
            cat = "simulation"
        elif "benchmark" in query.lower() or "基准" in query:
            cat = "benchmark"
        categories[cat].append(score.get("overall_score", 0))

    if categories:
        cat_scores = {k: mean(v) for k, v in categories.items() if v}
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(
            list(cat_scores.keys()), list(cat_scores.values()), color="#2ecc71"
        )
        ax.set_xlim(0, 5)
        ax.set_xlabel("Average Score")
        ax.set_title("Score by Query Category")
        for bar, val in zip(bars, cat_scores.values()):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
            )
        plt.tight_layout()
        plt.savefig(figures_dir / "category_scores.png", dpi=150)
        plt.close()

    print(f"Saved charts to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估 Agent + 学术搜索问答质量")
    parser.add_argument("--queries", required=True, help="学术搜索查询数据JSONL文件")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    args = parser.parse_args()

    # 加载配置
    config = load_config()
    judge_config = config.get("judge", {})

    # 加载完整配置（包含模型配置）
    full_config = load_config_full()

    # 初始化 Agent LLM (从根级别读取)
    agent_base_url = full_config.get("model_url", "https://api.deepseek.com")
    agent_model = full_config.get("llm", "deepseek-chat")
    agent_api_key = full_config.get("api_key", "")

    # 创建 Agent
    print(f"Creating agent with model: {agent_model} @ {agent_base_url}")
    agent = create_academic_agent(agent_base_url, agent_model, agent_api_key)

    # 初始化 LLM Judge
    llm = ChatOpenAI(
        model=judge_config.get("model", "deepseek-chat"),
        base_url=judge_config.get("api_base", "https://api.deepseek.com"),
        api_key=judge_config.get("api_key", ""),
        timeout=judge_config.get("timeout", 300),
        max_retries=judge_config.get("max_retries", 3),
    )

    # 加载查询
    queries = load_queries(Path(args.queries))
    print(
        f"Using Judge: {judge_config.get('model', 'deepseek-chat')} @ {judge_config.get('api_base', 'https://api.deepseek.com')}"
    )

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评估
    model_name = agent_model
    results = asyncio.run(
        evaluate_academic_agent(
            queries, agent, llm, model_name, out_dir, delay_between_queries=0
        )
    )

    # 生成统计
    scores = [r.get("score", {}) for r in results if r.get("score")]
    valid_scores = [s for s in scores if s.get("overall_score", 0) > 0]

    if valid_scores:
        summary = {
            "total": len(queries),
            "evaluated": len(valid_scores),
            "avg_relevance": mean([s.get("relevance", 0) for s in valid_scores]),
            "avg_accuracy": mean([s.get("accuracy", 0) for s in valid_scores]),
            "avg_completeness": mean([s.get("completeness", 0) for s in valid_scores]),
            "avg_citation": mean([s.get("citation", 0) for s in valid_scores]),
            "avg_overall_score": mean(
                [s.get("overall_score", 0) for s in valid_scores]
            ),
        }
    else:
        summary = {"total": len(queries), "evaluated": 0}

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total queries: {summary['total']}")
    print(f"Evaluated: {summary.get('evaluated', 0)}")
    if valid_scores:
        print(f"Avg relevance: {summary['avg_relevance']:.2f}")
        print(f"Avg accuracy: {summary['avg_accuracy']:.2f}")
        print(f"Avg completeness: {summary['avg_completeness']:.2f}")
        print(f"Avg citation: {summary['avg_citation']:.2f}")
        print(f"Avg overall score: {summary['avg_overall_score']:.2f}")

    # 生成图表
    generate_charts(results, out_dir)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
