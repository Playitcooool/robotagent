import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

_cached_subagents: List[Any] | None = None
_subagent_lock = asyncio.Lock()

# Collection runtime behavior (intended for single-trajectory isolation)
REBUILD_AGENT_EVERY = 1
CLEANUP_SIMULATION_PER_ATTEMPT = True
MCP_CLEANUP_URL = "http://localhost:8001/mcp"


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

    async def chat(self, system_prompt: str, user_content: str) -> str:
        from deepagents import create_deep_agent
        from langchain.agents.middleware import ToolRetryMiddleware
        from langchain_openai import ChatOpenAI

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


def load_prompts(path: str, max_prompts: int) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            prompts.append(text)
            if max_prompts > 0 and len(prompts) >= max_prompts:
                break
    return prompts


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def load_finished_keys(path: str) -> Set[Tuple[int, int]]:
    finished: Set[Tuple[int, int]] = set()
    if not os.path.exists(path):
        return finished
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                finished.add((int(item["prompt_id"]), int(item["attempt_id"])))
            except Exception:
                continue
    return finished


def extract_messages(agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for msg in agent_result.get("messages", []):
        cls_name = msg.__class__.__name__
        if cls_name not in {"HumanMessage", "AIMessage", "ToolMessage"}:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role in {"user", "assistant", "tool"}:
                    messages.append({"role": role, "content": content})
            continue

        if cls_name == "HumanMessage":
            role = "user"
        elif cls_name == "AIMessage":
            role = "assistant"
        else:
            role = "tool"

        item: Dict[str, Any] = {"role": role, "content": msg.content}

        if cls_name == "ToolMessage":
            name = getattr(msg, "name", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            if name:
                item["name"] = name
            if tool_call_id:
                item["tool_call_id"] = tool_call_id

        messages.append(item)
    return messages


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def load_memory_bank(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"meta": {}, "experiences": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        experiences = data.get("experiences", [])
        if not isinstance(experiences, list):
            experiences = []
        return {"meta": data.get("meta", {}), "experiences": experiences}
    except Exception:
        return {"meta": {}, "experiences": []}


def save_memory_bank(path: str, memory_bank: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory_bank, f, ensure_ascii=False, indent=2)


def render_memory_markdown(memory_bank: Dict[str, Any]) -> str:
    lines: List[str] = ["# Training-Free GRPO Online Memory", ""]
    experiences = memory_bank.get("experiences", [])
    for idx, item in enumerate(experiences, start=1):
        score = item.get("score", {})
        overall = score.get("overall_score", 0.0)
        lines.extend(
            [
                f"## Experience {idx}",
                f"- prompt_id: {item.get('prompt_id')}",
                f"- attempt_id: {item.get('attempt_id')}",
                f"- overall_score: {overall}",
                "",
                "### Summary",
                str(item.get("summary", {}).get("one_paragraph_summary", "")),
                "",
                "### Checklist",
            ]
        )
        for c in item.get("summary", {}).get("checklist", []):
            lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_online_system_prompt(
    base_prompt: str,
    experiences: List[Dict[str, Any]],
    max_experiences_in_prompt: int,
) -> str:
    if not experiences or max_experiences_in_prompt <= 0:
        return base_prompt
    tail = experiences[-max_experiences_in_prompt:]
    lines = [base_prompt.strip(), "", "【Online 经验库（按时间更新）】"]
    for idx, item in enumerate(tail, start=1):
        score = item.get("score", {})
        summary = item.get("summary", {})
        lines.append(
            f"{idx}. prompt_id={item.get('prompt_id')}, attempt_id={item.get('attempt_id')}, overall_score={score.get('overall_score', 0.0)}"
        )
        text = str(summary.get("one_paragraph_summary", "")).strip()
        if text:
            lines.append(f"   - 总结: {text}")
        checklist = summary.get("checklist", [])
        if isinstance(checklist, list):
            for c in checklist[:5]:
                lines.append(f"   - 检查项: {c}")
    lines.append("请优先遵守上述经验，避免重复失败模式。")
    return "\n".join(lines).strip()


def build_score_prompt(traj: Dict[str, Any]) -> Dict[str, str]:
    rubric = (
        "你是严格评审。请根据任务完成度、正确性、清晰度、鲁棒性、冗余度对轨迹评分。"
        "返回严格 JSON，不要额外文本。"
    )
    user_content = {
        "instruction": "请评分并解释。",
        "trajectory": traj.get("messages", []),
        "output_schema": {
            "overall_score": "0-10 float",
            "task_completion": "0-10 float",
            "correctness": "0-10 float",
            "clarity": "0-10 float",
            "robustness": "0-10 float",
            "conciseness": "0-10 float",
            "pros": ["string"],
            "cons": ["string"],
            "brief_reason": "string",
        },
    }
    return {"system": rubric, "user": json.dumps(user_content, ensure_ascii=False)}


def build_single_experience_prompt(
    trajectory: Dict[str, Any], score: Dict[str, Any]
) -> Dict[str, str]:
    system = (
        "你是经验提炼器。基于单条轨迹及其评分，提炼下一轮可执行经验。"
        "仅返回严格 JSON，不要额外文本。"
    )
    payload = {
        "task": "提炼单条经验并用于下一轮 system prompt",
        "trajectory": {
            "prompt": trajectory.get("prompt", ""),
            "response": trajectory.get("response", ""),
            "messages": trajectory.get("messages", []),
        },
        "score": score,
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


async def score_trajectory(
    judge_client: DeepAgentClient, trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = build_score_prompt(trajectory)
    raw = await judge_client.chat(prompt["system"], prompt["user"])
    parsed = extract_first_json_object(raw)
    if parsed is None:
        parsed = {
            "overall_score": 0.0,
            "task_completion": 0.0,
            "correctness": 0.0,
            "clarity": 0.0,
            "robustness": 0.0,
            "conciseness": 0.0,
            "pros": [],
            "cons": ["judge_output_not_json"],
            "brief_reason": raw[:300],
        }
    return parsed


async def summarize_single_experience(
    summary_client: DeepAgentClient,
    trajectory: Dict[str, Any],
    score: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_single_experience_prompt(trajectory=trajectory, score=score)
    raw = await summary_client.chat(prompt["system"], prompt["user"])
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
    return parsed


def resolve_base_system_prompt(system_prompt: str) -> str:
    if system_prompt.strip():
        return system_prompt.strip()
    from prompts import MainAgentPrompt

    return MainAgentPrompt.SYSTEM_PROMPT


async def collect_trajectories_online(
    agent_builder: Callable[[str], Awaitable[Any]],
    prompts: List[str],
    output_path: str,
    score_path: str,
    memory_json_path: str,
    memory_md_path: str,
    samples_per_prompt: int,
    rebuild_agent_every: int,
    base_system_prompt: str,
    judge_client: DeepAgentClient,
    summary_client: DeepAgentClient,
    max_experiences_in_prompt: int,
    cleanup_after_attempt: Callable[[], Awaitable[None]] | None = None,
) -> None:
    finished_keys = load_finished_keys(output_path)
    scored_keys = load_finished_keys(score_path)
    memory_bank = load_memory_bank(memory_json_path)
    if "experiences" not in memory_bank or not isinstance(
        memory_bank["experiences"], list
    ):
        memory_bank["experiences"] = []
    print(f"[collect] existing trajectories: {len(finished_keys)}")
    print(f"[collect] existing scores: {len(scored_keys)}")
    print(f"[collect] existing experiences: {len(memory_bank['experiences'])}")

    agent = None
    attempt_since_rebuild = 0

    for prompt_id, prompt in enumerate(prompts):
        for attempt_id in range(samples_per_prompt):
            key = (prompt_id, attempt_id)
            if key in finished_keys:
                continue

            current_system_prompt = build_online_system_prompt(
                base_prompt=base_system_prompt,
                experiences=memory_bank["experiences"],
                max_experiences_in_prompt=max_experiences_in_prompt,
            )

            try:
                if (
                    agent is None
                    or rebuild_agent_every > 0
                    and attempt_since_rebuild >= rebuild_agent_every
                ):
                    agent = await agent_builder(current_system_prompt)
                    attempt_since_rebuild = 0
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                attempt_since_rebuild += 1
            except Exception as e:
                print(
                    f"[collect] failed prompt={prompt_id} attempt={attempt_id}: {e}"
                )
                continue
            finally:
                if cleanup_after_attempt is not None:
                    try:
                        await cleanup_after_attempt()
                    except Exception as e:
                        print(f"[collect] cleanup_after_attempt failed: {e}")

            trajectory_messages = extract_messages(result)
            response = ""
            for item in reversed(trajectory_messages):
                if item.get("role") == "assistant":
                    response = str(item.get("content", ""))
                    break

            trajectory = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "prompt": prompt,
                "messages": trajectory_messages,
                "response": response,
                "meta": {
                    "method": "deep_agent_online_memory",
                    "created_at": int(time.time()),
                },
            }
            append_jsonl(output_path, trajectory)
            finished_keys.add(key)
            print(f"[collect] saved prompt={prompt_id} attempt={attempt_id}")

            if key in scored_keys:
                continue

            score = await score_trajectory(judge_client=judge_client, trajectory=trajectory)
            score_record: Dict[str, Any] = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "prompt": prompt,
                "messages": trajectory_messages,
                "response": response,
                "score": score,
                "meta": {
                    "model": judge_client.model,
                    "created_at": int(time.time()),
                    "method": "deep_agent_online_memory_judge",
                    "judge_temperature": judge_client.temperature,
                },
            }
            append_jsonl(score_path, score_record)
            scored_keys.add(key)
            print(f"[score] saved prompt={prompt_id} attempt={attempt_id}")

            summary = await summarize_single_experience(
                summary_client=summary_client, trajectory=trajectory, score=score
            )
            experience_item = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "score": score,
                "summary": summary,
                "created_at": int(time.time()),
            }
            memory_bank["meta"] = {
                "method": "training_free_grpo_online_memory",
                "updated_at": int(time.time()),
                "source_trajectory_file": output_path,
                "source_score_file": score_path,
            }
            memory_bank["experiences"].append(experience_item)
            save_memory_bank(memory_json_path, memory_bank)
            with open(memory_md_path, "w", encoding="utf-8") as f:
                f.write(render_memory_markdown(memory_bank))
            print(
                f"[memory] appended experience prompt={prompt_id} attempt={attempt_id}; total={len(memory_bank['experiences'])}"
            )


async def build_agent(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    max_retries: int,
    backoff_factor: float,
    initial_delay: float,
) -> Any:
    from deepagents import create_deep_agent
    from langchain.agents.middleware import ToolRetryMiddleware
    from langchain_openai import ChatOpenAI
    from tools.SubAgentTool import init_subagents

    async with _subagent_lock:
        global _cached_subagents
        if _cached_subagents is None:
            _cached_subagents = list(await init_subagents())
    subagents = _cached_subagents

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )

    return create_deep_agent(
        model=chat,
        tools=[],
        system_prompt=system_prompt,
        subagents=subagents,
        middleware=[
            ToolRetryMiddleware(
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                initial_delay=initial_delay,
            )
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect trajectories with online experience memory: "
            "collect -> score -> summarize-one -> append memory -> next round."
        )
    )
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1")
    parser.add_argument(
        "--model", type=str, default="lmstudio-community-qwen3-4b-instruct-2507-mlx"
    )
    parser.add_argument("--api_key", type=str, default="no_need")
    parser.add_argument("--prompts", type=str, default="SFT/data.txt")
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/training_free_grpo/trajectories.jsonl",
    )
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
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--samples_per_prompt", type=int, default=3)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    parser.add_argument("--initial_delay", type=float, default=1.0)
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--summary_model", type=str, default="")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--summary_temperature", type=float, default=0.2)
    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--summary_max_tokens", type=int, default=1024)
    parser.add_argument("--max_experiences_in_prompt", type=int, default=20)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    for path in [
        os.path.dirname(args.output_path),
        os.path.dirname(args.score_path),
        os.path.dirname(args.memory_json_path),
        os.path.dirname(args.memory_md_path),
    ]:
        if path:
            ensure_dir(path)

    prompts = load_prompts(args.prompts, args.max_prompts)
    base_system_prompt = resolve_base_system_prompt(args.system_prompt)

    async def agent_builder(system_prompt: str) -> Any:
        return await build_agent(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            system_prompt=system_prompt,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
            initial_delay=args.initial_delay,
        )

    judge_model = args.judge_model.strip() or args.model
    summary_model = args.summary_model.strip() or args.model
    judge_client = DeepAgentClient(
        base_url=args.base_url,
        model=judge_model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        initial_delay=args.initial_delay,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )
    summary_client = DeepAgentClient(
        base_url=args.base_url,
        model=summary_model,
        api_key=args.api_key,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        initial_delay=args.initial_delay,
        temperature=args.summary_temperature,
        max_tokens=args.summary_max_tokens,
    )

    cleanup_hook = None
    if CLEANUP_SIMULATION_PER_ATTEMPT:
        from fastmcp import Client

        mcp_client = Client(MCP_CLEANUP_URL)

        async def cleanup_hook() -> None:
            await mcp_client.call_tool("cleanup_simulation_tool")

        async with mcp_client:
            await collect_trajectories_online(
                agent_builder=agent_builder,
                prompts=prompts,
                output_path=args.output_path,
                score_path=args.score_path,
                memory_json_path=args.memory_json_path,
                memory_md_path=args.memory_md_path,
                samples_per_prompt=args.samples_per_prompt,
                rebuild_agent_every=REBUILD_AGENT_EVERY,
                base_system_prompt=base_system_prompt,
                judge_client=judge_client,
                summary_client=summary_client,
                max_experiences_in_prompt=args.max_experiences_in_prompt,
                cleanup_after_attempt=cleanup_hook,
            )
    else:
        await collect_trajectories_online(
            agent_builder=agent_builder,
            prompts=prompts,
            output_path=args.output_path,
            score_path=args.score_path,
            memory_json_path=args.memory_json_path,
            memory_md_path=args.memory_md_path,
            samples_per_prompt=args.samples_per_prompt,
            rebuild_agent_every=REBUILD_AGENT_EVERY,
            base_system_prompt=base_system_prompt,
            judge_client=judge_client,
            summary_client=summary_client,
            max_experiences_in_prompt=args.max_experiences_in_prompt,
            cleanup_after_attempt=cleanup_hook,
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
