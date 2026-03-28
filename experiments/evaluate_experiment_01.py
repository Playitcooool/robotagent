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

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,host.docker.internal")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,host.docker.internal")

import argparse
import json
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
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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
AGENT_SYSTEM_PROMPT = """你是一个专业的学术问答助手，专门回答机器人领域的问题。

你可以通过调用 search 工具搜索信息来回答问题。
该工具同时搜索网页（Tavily）和学术论文（OpenAlex + arXiv），并自动去重合并结果。

请严格按照以下步骤回答问题：
1. 分析用户问题，确定需要搜索的关键词
2. 调用 search 工具搜索相关信息
3. 根据搜索结果整理并回答用户问题

【强制引用规则——必须遵守】
- 回答中的每一条事实性陈述，都必须标注来源
- 引用格式：在陈述末尾加 [来源序号]
- 在回答末尾必须包含完整的参考文献列表，格式如下：
  - 论文：[编号] 作者, 标题, 年份, 期刊/会议, URL
  - 网页：[编号] 标题, URL
- 禁止在没有任何来源支撑的情况下给出事实性陈述
- 不确定的内容必须明确写"该信息未经核实"或"搜索结果未提供此信息"，不得臆测

【准确性规则】
- 只引用搜索结果中明确包含的信息
- 不要基于部分信息进行推断或扩展
- 如果搜索结果不足以回答问题，明确说明"搜索结果不足以完全回答此问题"

注意：
- 每个问题最多搜索 1~2 次，避免过度搜索
- 搜索 1 次后已有足够信息就直接回答，不要反复搜索
- 只在搜索结果明显不足或与问题不相关时才重新搜索
- 回答控制在 300 字以内（不含参考文献列表）"""

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
    agent = create_agent(
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


def extract_json(text: str) -> dict | None:
    """从文本中提取第一个有效的 JSON 对象（支持嵌套）"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


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

            # 提取JSON（支持嵌套）
            parsed = extract_json(content)
            if parsed is None:
                # 尝试直接解析整个content
                parsed = json.loads(content.strip())
            # 确保有所有必要字段
            parsed.setdefault("relevance", 0)
            parsed.setdefault("accuracy", 0)
            parsed.setdefault("completeness", 0)
            parsed.setdefault("citation", 0)
            parsed.setdefault("overall_score", 0)
            parsed.setdefault("pros", [])
            parsed.setdefault("cons", [])
            parsed.setdefault("brief_reason", "")
            return parsed
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


def load_existing_results(out_dir: Path) -> tuple[list, set]:
    """加载已存在的结果，返回 (results列表, 已完成query_id集合)"""
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
                completed_ids.add(r.get("query_id", ""))
            except Exception:
                continue
    print(f"Loaded {len(existing_results)} existing results from {details_file}")
    return existing_results, completed_ids


async def evaluate_academic_agent(
    queries: list,
    agent,
    llm,
    out_dir: Path,
    delay_between_queries: float = 3.0,
):
    """评估 Agent + 学术搜索系统

    Args:
        delay_between_queries: 每次查询之间的延迟（秒），用于避免 API 限流
    """
    import asyncio

    # 加载已存在的结果，支持断点续跑
    results, completed_ids = load_existing_results(out_dir)
    start_index = len(results)

    total = len(queries)
    print(
        f"Total queries: {total}, Already completed: {len(completed_ids)}, Remaining: {total - start_index}"
    )

    for i, q in enumerate(queries, 1):
        query_id = q.get("id", f"query_{i}")

        # 跳过已完成的query
        if query_id in completed_ids:
            print(f"[{i}/{total}] Skipping {query_id} (already completed)")
            continue

        query_text = q.get("query", "")

        print(f"[{i}/{total}] Evaluating {query_id}: {query_text[:30]}...")

        # 调用 Agent
        answer = await call_agent(agent, query_text)

        if not answer:
            print(f"[WARN] No answer for {query_id}, skipping...")
            record = {
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
        else:
            # 调用 Judge 评估
            score = call_judge(llm, query_text, answer)
            record = {
                "query_id": query_id,
                "query": query_text,
                "answer": answer,
                "score": score,
            }

        results.append(record)
        completed_ids.add(query_id)

        # 保存进度（追加模式，避免中断丢失）
        with (out_dir / "details.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 每次查询结束后等待，避免触发 API 限流
        if i < total:
            print(f"  Waiting {delay_between_queries}s before next query...")
            await asyncio.sleep(delay_between_queries)

    return results


def generate_charts(results: list, out_dir: Path):
    """生成可视化图表"""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 提取分数
    scores = [r.get("score", {}) for r in results if r.get("score")]
    if not scores:
        return

    # 1. 评分维度柱状图（百分比刻度）
    dimensions = ["relevance", "accuracy", "completeness", "citation", "overall_score"]
    avg_scores = {d: mean([s.get(d, 0) for s in scores]) for d in dimensions}
    pct_scores = {d: (v / 5.0 * 100) for d, v in avg_scores.items()}

    _, ax = plt.subplots(figsize=(12, 6))
    labels = [
        "Relevance\n(相关度)",
        "Accuracy\n(准确性)",
        "Completeness\n(完整性)",
        "Citation\n(引用率)",
        "Overall\n(综合得分)",
    ]
    vals = [pct_scores[d] for d in dimensions]
    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Agent + Academic Search Quality (%)")
    for bar, val, orig in zip(bars, vals, [avg_scores[d] for d in dimensions]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%\n({orig:.2f}/5)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80% target")
    ax.axhline(y=60, color="orange", linestyle="--", alpha=0.5, label="60% baseline")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "score_bars.png", dpi=150)
    plt.close()

    # 2. 得分分布直方图
    overall_scores = [
        s.get("overall_score", 0) for s in scores if s.get("overall_score", 0) > 0
    ]
    if overall_scores:
        _, ax = plt.subplots(figsize=(10, 6))
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
        _, ax = plt.subplots(figsize=(8, 5))
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
    results = asyncio.run(
        evaluate_academic_agent(queries, agent, llm, out_dir, delay_between_queries=0)
    )

    # 生成统计
    scores = [r.get("score", {}) for r in results if r.get("score")]
    valid_scores = [s for s in scores if s.get("overall_score", 0) > 0]

    if valid_scores:
        avg_r = mean([s.get("relevance", 0) for s in valid_scores])
        avg_a = mean([s.get("accuracy", 0) for s in valid_scores])
        avg_c = mean([s.get("completeness", 0) for s in valid_scores])
        avg_ci = mean([s.get("citation", 0) for s in valid_scores])
        avg_o = mean([s.get("overall_score", 0) for s in valid_scores])
        summary = {
            "total": len(queries),
            "evaluated": len(valid_scores),
            "avg_relevance": avg_r,
            "avg_accuracy": avg_a,
            "avg_completeness": avg_c,
            "avg_citation": avg_ci,
            "avg_overall_score": avg_o,
            "avg_relevance_pct": round(avg_r / 5.0 * 100, 1),
            "avg_accuracy_pct": round(avg_a / 5.0 * 100, 1),
            "avg_completeness_pct": round(avg_c / 5.0 * 100, 1),
            "avg_citation_pct": round(avg_ci / 5.0 * 100, 1),
            "avg_overall_score_pct": round(avg_o / 5.0 * 100, 1),
        }
    else:
        summary = {"total": len(queries), "evaluated": 0}

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total queries: {summary['total']}")
    print(f"Evaluated: {summary.get('evaluated', 0)}")
    if valid_scores:
        print(
            f"Avg relevance:   {summary['avg_relevance']:.2f}/5  ({summary['avg_relevance_pct']:.1f}%)"
        )
        print(
            f"Avg accuracy:    {summary['avg_accuracy']:.2f}/5  ({summary['avg_accuracy_pct']:.1f}%)"
        )
        print(
            f"Avg completeness:{summary['avg_completeness']:.2f}/5  ({summary['avg_completeness_pct']:.1f}%)"
        )
        print(
            f"Avg citation:    {summary['avg_citation']:.2f}/5  ({summary['avg_citation_pct']:.1f}%)"
        )
        print(
            f"Avg overall:     {summary['avg_overall_score']:.2f}/5  ({summary['avg_overall_score_pct']:.1f}%)"
        )

    # 生成图表
    generate_charts(results, out_dir)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
