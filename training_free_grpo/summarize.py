import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from deepagents import create_deep_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI


@dataclass
class DeepAgentClient:
    base_url: str
    model: str
    api_key: str
    max_retries: int
    backoff_factor: float
    initial_delay: float
    temperature: float
    max_tokens: int

    async def chat(
        self,
        system_prompt: str,
        user_content: str,
    ) -> str:
        chat = ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        agent = create_deep_agent(
            model=chat,
            tools=[],
            system_prompt=system_prompt,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=self.max_retries,
                    backoff_factor=self.backoff_factor,
                    initial_delay=self.initial_delay,
                )
            ],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]}
        )

        for msg in reversed(result.get("messages", [])):
            cls_name = msg.__class__.__name__
            if cls_name == "AIMessage":
                return str(msg.content).strip()
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        raise RuntimeError("No assistant response in deepagent result")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def build_summary_prompt(
    agent_name: str,
    top: List[Dict[str, Any]],
    bottom: List[Dict[str, Any]],
) -> Dict[str, str]:
    agent_focus_map = {
        "main_agent": "主Agent的任务拆解、路由决策、结果汇总策略",
        "simulation_agent": "仿真Agent的参数设置、工具调用顺序、状态校验策略",
        "analysis_agent": "分析Agent的数据清洗、统计解释、可视化与结论表达策略",
    }
    focus = agent_focus_map.get(agent_name, "通用智能体策略")
    system = (
        "你是经验提炼器。基于高分/低分样本，提炼可执行策略。"
        f"本轮必须聚焦：{focus}。"
        "仅返回严格 JSON，不要额外文本。"
    )
    payload = {
        "task": "提炼可迁移经验",
        "agent_name": agent_name,
        "top_examples": top,
        "bottom_examples": bottom,
        "output_schema": {
            "principles": ["string"],
            "dos": ["string"],
            "donts": ["string"],
            "failure_patterns": ["string"],
            "checklist": ["string"],
            "one_paragraph_summary": "string",
        },
    }
    return {"system": system, "user": json.dumps(payload, ensure_ascii=False)}


def pick_top_bottom(
    scores: List[Dict[str, Any]], top_k: int, bottom_k: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def get_score(x: Dict[str, Any]) -> float:
        try:
            return float(x.get("score", {}).get("overall_score", 0.0))
        except Exception:
            return 0.0

    ranked = sorted(scores, key=get_score, reverse=True)
    top = ranked[: max(0, top_k)]
    bottom = ranked[-max(0, bottom_k) :] if bottom_k > 0 else []
    return top, bottom


def _contains_any(text: str, terms: List[str]) -> bool:
    lower = text.lower()
    return any(t in lower for t in terms)


def detect_agent_bucket(score_item: Dict[str, Any]) -> str:
    messages = score_item.get("messages", [])
    response = str(score_item.get("response", ""))
    prompt = str(score_item.get("prompt", ""))
    whole_text = f"{response}\n{prompt}"

    simulation_terms = [
        "simulation",
        "pybullet",
        "path_planning",
        "push_cube_step",
        "grab_and_place",
        "cleanup_simulation",
        "simulator",
    ]
    analysis_terms = [
        "analysis",
        "plot",
        "histogram",
        "correlation",
        "describe_stats",
        "summarize_csv",
        "time_series",
        "统计",
        "分析",
        "可视化",
    ]

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        name = str(msg.get("name", "")).lower()
        content = str(msg.get("content", "")).lower()
        merged = f"{name}\n{content}"
        if _contains_any(merged, simulation_terms):
            return "simulation_agent"
        if _contains_any(merged, analysis_terms):
            return "analysis_agent"

    if _contains_any(whole_text, simulation_terms):
        return "simulation_agent"
    if _contains_any(whole_text, analysis_terms):
        return "analysis_agent"
    return "main_agent"


def split_scores_by_agent(scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "main_agent": [],
        "simulation_agent": [],
        "analysis_agent": [],
    }
    for item in scores:
        bucket = detect_agent_bucket(item)
        grouped[bucket].append(item)
    return grouped


async def summarize_experience(
    client: DeepAgentClient,
    score_path: str,
    memory_json_path: str,
    memory_md_path: str,
    top_k: int,
    bottom_k: int,
) -> None:
    scores = load_jsonl(score_path)
    if not scores:
        raise RuntimeError("No score data found. Run score step first.")

    grouped = split_scores_by_agent(scores)
    per_agent_experience: Dict[str, Dict[str, Any]] = {}
    per_agent_stats: Dict[str, Dict[str, int]] = {}

    for agent_name, agent_scores in grouped.items():
        if not agent_scores:
            per_agent_experience[agent_name] = {
                "principles": [],
                "dos": [],
                "donts": [],
                "failure_patterns": ["no_samples_for_agent"],
                "checklist": [],
                "one_paragraph_summary": "No trajectories available for this agent.",
            }
            per_agent_stats[agent_name] = {"total": 0, "top_used": 0, "bottom_used": 0}
            continue

        top, bottom = pick_top_bottom(agent_scores, top_k=top_k, bottom_k=bottom_k)
        prompt = build_summary_prompt(agent_name=agent_name, top=top, bottom=bottom)
        raw = await client.chat(prompt["system"], prompt["user"])
        parsed = extract_first_json_object(raw)
        if parsed is None:
            parsed = {
                "principles": [],
                "dos": [],
                "donts": [],
                "failure_patterns": ["summary_output_not_json"],
                "checklist": [],
                "one_paragraph_summary": raw[:800],
            }
        per_agent_experience[agent_name] = parsed
        per_agent_stats[agent_name] = {
            "total": len(agent_scores),
            "top_used": len(top),
            "bottom_used": len(bottom),
        }

    external_memory = {
        "meta": {
            "model": client.model,
            "created_at": int(time.time()),
            "top_k": top_k,
            "bottom_k": bottom_k,
            "source_score_file": score_path,
            "method": "training_free_grpo_deep_agent",
        },
        "per_agent_stats": per_agent_stats,
        "experience": per_agent_experience,
    }

    with open(memory_json_path, "w", encoding="utf-8") as f:
        json.dump(external_memory, f, ensure_ascii=False, indent=2)

    lines = [
        "# Training-Free GRPO External Memory",
        "",
        f"- model: {client.model}",
        f"- created_at: {external_memory['meta']['created_at']}",
        f"- top_k: {top_k}",
        f"- bottom_k: {bottom_k}",
        "",
        "## Agent Sample Stats",
    ]
    for agent_name, stat in per_agent_stats.items():
        lines.append(
            f"- {agent_name}: total={stat.get('total', 0)}, top_used={stat.get('top_used', 0)}, bottom_used={stat.get('bottom_used', 0)}"
        )

    for agent_name in ["main_agent", "simulation_agent", "analysis_agent"]:
        exp = per_agent_experience.get(agent_name, {})
        lines.extend(
            [
                "",
                f"## {agent_name}",
                "",
                "### Summary",
                str(exp.get("one_paragraph_summary", "")),
                "",
                "### Principles",
            ]
        )
        for x in exp.get("principles", []):
            lines.append(f"- {x}")
        lines.extend(["", "### Dos"])
        for x in exp.get("dos", []):
            lines.append(f"- {x}")
        lines.extend(["", "### Donts"])
        for x in exp.get("donts", []):
            lines.append(f"- {x}")
        lines.extend(["", "### Failure Patterns"])
        for x in exp.get("failure_patterns", []):
            lines.append(f"- {x}")
        lines.extend(["", "### Checklist"])
        for x in exp.get("checklist", []):
            lines.append(f"- {x}")

    with open(memory_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[summarize] external memory saved to {memory_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scored trajectories only.")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument(
        "--model", type=str, default="lmstudio-community-qwen3-4b-instruct-2507-mlx"
    )
    parser.add_argument("--api_key", type=str, default="no_need")
    parser.add_argument(
        "--score_path",
        type=str,
        default="output/training_free_grpo/trajectory_scores.jsonl",
    )
    parser.add_argument(
        "--memory_json_path",
        type=str,
        default="output/training_free_grpo/external_memory.json",
    )
    parser.add_argument(
        "--memory_md_path",
        type=str,
        default="output/training_free_grpo/external_memory.md",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bottom_k", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    parser.add_argument("--initial_delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    json_output_dir = os.path.dirname(args.memory_json_path)
    md_output_dir = os.path.dirname(args.memory_md_path)
    if json_output_dir:
        ensure_dir(json_output_dir)
    if md_output_dir:
        ensure_dir(md_output_dir)

    client = DeepAgentClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        initial_delay=args.initial_delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    await summarize_experience(
        client=client,
        score_path=args.score_path,
        memory_json_path=args.memory_json_path,
        memory_md_path=args.memory_md_path,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
