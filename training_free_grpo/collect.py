import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,host.docker.internal")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,host.docker.internal")

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import yaml

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from training_free_grpo.experience_tools import (
    JUDGE_EXPERIENCE_TOOLS,
    build_judge_agent,
    score_only,
    grpo_summarize_and_update_memory,
)

# Collection runtime behavior (intended for single-trajectory isolation)
REBUILD_AGENT_EVERY = 1  # 每N次attempt重建一次agent，确保每次attempt独立
CLEANUP_SIMULATION_PER_ATTEMPT = False  # False = 保持仿真环境持久化，init一次后复用
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DEFAULT_ATTEMPT_TIMEOUT_S = 2000
DEFAULT_CLEANUP_TIMEOUT_S = 60.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


async def maybe_wait_for(awaitable: Awaitable[Any], timeout_s: Optional[float]) -> Any:
    if timeout_s is None or timeout_s <= 0:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout_s)


async def stream_with_timeout(agent, input_dict, timeout_s, print_prefix="[agent]"):
    """Run agent.astream with timeout, yielding chunks and printing progress."""
    result = None
    try:
        async for chunk in asyncio.wait_for(
            agent.astream(input_dict), timeout=timeout_s
        ):
            print(f"{print_prefix} stream: {type(chunk).__name__}: {str(chunk)[:300]}")
            result = chunk
        return result
    except asyncio.TimeoutError:
        print(f"{print_prefix} TIMEOUT after {timeout_s}s")
        raise


def normalize_timeout(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    return v if v > 0 else None


def is_stream_error(err: Exception) -> bool:
    text = str(err).lower()
    return (
        "sse" in text
        or "streamable_http" in text
        or "brokenresourceerror" in text
        or "closedresourceerror" in text
        or "connection" in text
        and "closed" in text
    )


def load_prompts(path: str, max_prompts: int) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
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
    """提取消息，并识别每个消息来自哪个 agent"""
    messages: List[Dict[str, Any]] = []
    current_agent = "main"  # 默认是主 agent

    for msg in agent_result.get("messages", []):
        cls_name = msg.__class__.__name__

        # 检查是否是子 agent 的响应
        if cls_name == "AIMessage":
            # 检查消息是否有 subagent 标识
            msg_dict = (
                msg
                if isinstance(msg, dict)
                else vars(msg) if hasattr(msg, "__dict__") else {}
            )
            # 通过 tool_calls 或 content 判断是否是调用子 agent
            # 子 agent 响应通常包含 agent 名称

        if cls_name not in {"HumanMessage", "AIMessage", "ToolMessage"}:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role in {"user", "assistant", "tool"}:
                    messages.append(
                        {"role": role, "content": content, "agent": current_agent}
                    )
            continue

        if cls_name == "HumanMessage":
            role = "user"
        elif cls_name == "AIMessage":
            role = "assistant"
            # 检查是否是调用子 agent
            msg_dict = msg if isinstance(msg, dict) else {}
            # 尝试从 tool_calls 中识别子 agent
            tool_calls = msg_dict.get("tool_calls", []) or getattr(
                msg, "tool_calls", []
            )
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_func = tc.get("function", {})
                    func_name = (
                        tc_func.get("name", "") if isinstance(tc_func, dict) else ""
                    )
                    # 根据工具名判断是哪个 agent
                    if func_name == "data_analyzer":
                        current_agent = "data-analyzer"
                    elif func_name == "simulator":
                        current_agent = "simulator"
        else:
            role = "tool"

        item: Dict[str, Any] = {
            "role": role,
            "content": msg.content,
            "agent": current_agent,
        }

        if cls_name == "ToolMessage":
            name = getattr(msg, "name", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            if name:
                item["name"] = name
                # 根据工具所属判断 agent
                if name in {"analyze_trajectory", "analyze_code", "analyze_experiment"}:
                    item["agent"] = "data-analyzer"
                elif name in {
                    "initialize_simulation",
                    "step_simulation",
                    "reset_simulation",
                    "get_observation",
                }:
                    item["agent"] = "simulator"
                else:
                    item["agent"] = "main"
            if tool_call_id:
                item["tool_call_id"] = tool_call_id

        messages.append(item)
    return messages


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
    """Render memory bank as markdown (legacy format, for backward compat)."""
    lines: List[str] = ["# Training-Free GRPO Online Memory", ""]
    experiences = memory_bank.get("experiences", [])
    for idx, item in enumerate(experiences, start=1):
        score = item.get("score", 0.0)
        lines.extend(
            [
                f"## Experience {idx}",
                f"- id: {item.get('id', 'N/A')}",
                f"- prompt_id: {item.get('prompt_id')}",
                f"- overall_score: {score}",
                "",
                "### Summary",
                str(item.get("summary", "")),
                "",
                "### Principles",
            ]
        )
        for p in item.get("principles", []):
            lines.append(f"- {p}")
        lines.extend(["", "### Dos"])
        for d in item.get("dos", []):
            lines.append(f"- {d}")
        lines.extend(["", "### Donts"])
        for d in item.get("donts", []):
            lines.append(f"- {d}")
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
    lines = [base_prompt.strip()]
    for idx, item in enumerate(tail, start=1):
        score = item.get("score", 0.0)
        lines.append(
            f"{idx}. prompt_id={item.get('prompt_id')}, id={item.get('id', 'N/A')}, score={score}"
        )
        summary = str(item.get("summary", "")).strip()
        if summary:
            lines.append(f"   - 总结: {summary}")
        for p in item.get("principles", [])[:5]:
            lines.append(f"   - 原则: {p}")
    lines.append("请优先遵守上述经验，避免重复失败模式。")
    return "\n".join(lines).strip()


def resolve_base_system_prompt(system_prompt: str) -> str:
    if system_prompt.strip():
        return system_prompt.strip()
    from prompts import MainAgentPrompt

    return MainAgentPrompt.SYSTEM_PROMPT


async def collect_trajectories_online(
    agent_builder: Callable[[str], Awaitable[Any]],
    prompts: List[str],
    output_path: str,
    mirror_output_path: Optional[str],
    score_path: str,
    memory_json_path: str,
    memory_md_path: str,
    samples_per_prompt: int,
    rebuild_agent_every: int,
    base_system_prompt: str,
    judge_agent: Any,
    judge_model: Any,
    max_experiences_in_prompt: int,
    attempt_timeout_s: Optional[float],
    cleanup_timeout_s: Optional[float],
    attempt_retries: int,
    attempt_retry_delay_s: float,
    attempt_retry_backoff: float,
    cleanup_after_attempt: Callable[[], Awaitable[None]] | None = None,
    init_before_attempt: Callable[[], Awaitable[None]] | None = None,
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
        pending_for_prompt: List[Dict[str, Any]] = []

        for attempt_id in range(samples_per_prompt):
            key = (prompt_id, attempt_id)
            if key in finished_keys:
                continue

            current_system_prompt = build_online_system_prompt(
                base_prompt=base_system_prompt,
                experiences=memory_bank["experiences"],
                max_experiences_in_prompt=max_experiences_in_prompt,
            )

            result = None
            retry_delay = max(0.0, attempt_retry_delay_s)
            for retry_idx in range(max(0, attempt_retries) + 1):
                if init_before_attempt is not None:
                    try:
                        print(
                            f"[collect] init_hook start prompt={prompt_id} attempt={attempt_id}"
                        )
                        await maybe_wait_for(init_before_attempt(), cleanup_timeout_s)
                        print(
                            f"[collect] init_hook done prompt={prompt_id} attempt={attempt_id}"
                        )
                    except Exception as e:
                        print(f"[collect] init_before_attempt failed: {e}")

                try:
                    if (
                        agent is None
                        or rebuild_agent_every > 0
                        and attempt_since_rebuild >= rebuild_agent_every
                    ):
                        agent = await agent_builder(
                            system_prompt=current_system_prompt,
                            experiences=memory_bank["experiences"],
                        )
                        attempt_since_rebuild = 0
                    print(
                        f"[collect] agent.ainvoke start prompt={prompt_id} attempt={attempt_id}"
                    )
                    try:
                        result = await maybe_wait_for(
                            agent.ainvoke(
                                {"messages": [{"role": "user", "content": prompt}]}
                            ),
                            attempt_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        print(
                            f"[collect] timeout prompt={prompt_id} attempt={attempt_id}"
                        )
                        raise
                    print(
                        f"[collect] agent.ainvoke done prompt={prompt_id} attempt={attempt_id}"
                    )
                    attempt_since_rebuild += 1
                    break
                except asyncio.TimeoutError:
                    print(
                        f"[collect] timeout prompt={prompt_id} attempt={attempt_id} after {attempt_timeout_s}s"
                    )
                except Exception as e:
                    print(
                        f"[collect] failed prompt={prompt_id} attempt={attempt_id} retry={retry_idx}: {e}"
                    )
                    if is_stream_error(e):
                        from tools.SubAgentTool import reset_cached_mcp_tools

                        reset_cached_mcp_tools()
                    agent = None
                finally:
                    if cleanup_after_attempt is not None:
                        try:
                            print(
                                f"[collect] cleanup_hook start prompt={prompt_id} attempt={attempt_id}"
                            )
                            await maybe_wait_for(
                                cleanup_after_attempt(), cleanup_timeout_s
                            )
                            print(
                                f"[collect] cleanup_hook done prompt={prompt_id} attempt={attempt_id}"
                            )
                        except Exception as e:
                            print(f"[collect] cleanup_after_attempt failed: {e}")

                if retry_idx < max(0, attempt_retries):
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
                    retry_delay = (
                        retry_delay * attempt_retry_backoff
                        if attempt_retry_backoff > 0
                        else retry_delay
                    )

            if result is None:
                continue

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
            if mirror_output_path and mirror_output_path != output_path:
                append_jsonl(mirror_output_path, trajectory)
            finished_keys.add(key)
            print(f"[collect] saved prompt={prompt_id} attempt={attempt_id}")

            pending_for_prompt.append(trajectory)

        if not pending_for_prompt:
            continue

        # GRPO Step 1: Score all trajectories (no memory update yet)
        scored_this_round: List[Tuple[Dict[str, Any], dict]] = []
        for trajectory in pending_for_prompt:
            key = (trajectory["prompt_id"], trajectory["attempt_id"])
            if key in scored_keys:
                continue

            score = None
            retry_delay = max(0.0, attempt_retry_delay_s)
            for retry_idx in range(max(0, attempt_retries) + 1):
                try:
                    print(
                        f"[score] score_only start prompt={trajectory['prompt_id']} attempt={trajectory['attempt_id']}"
                    )
                    score = await maybe_wait_for(
                        score_only(
                            model=judge_model,
                            prompt_id=trajectory["prompt_id"],
                            attempt_id=trajectory["attempt_id"],
                            prompt=trajectory["prompt"],
                            messages=trajectory["messages"],
                            request_timeout_s=attempt_timeout_s,
                        ),
                        attempt_timeout_s,
                    )
                    print(
                        f"[score] score_only done prompt={trajectory['prompt_id']} attempt={trajectory['attempt_id']} score={score.get('overall_score', 'N/A')}"
                    )
                    break
                except asyncio.TimeoutError:
                    print(
                        f"[score] timeout prompt={trajectory['prompt_id']} attempt={trajectory['attempt_id']} after {attempt_timeout_s}s"
                    )
                except Exception as e:
                    print(
                        f"[score] failed prompt={trajectory['prompt_id']} attempt={trajectory['attempt_id']} retry={retry_idx}: {e}"
                    )
                    if is_stream_error(e):
                        from tools.SubAgentTool import reset_cached_mcp_tools

                        reset_cached_mcp_tools()

                if retry_idx < max(0, attempt_retries):
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
                    retry_delay = (
                        retry_delay * attempt_retry_backoff
                        if attempt_retry_backoff > 0
                        else retry_delay
                    )

            if score is not None:
                trajectory["score"] = score
                scored_this_round.append((trajectory, score))

        if not scored_this_round:
            print(f"[score] no valid scores for prompt={prompt_id}, skipping")
            continue

        # GRPO Step 2: Find best and worst by overall_score
        best_trajectory, best_score = max(
            scored_this_round, key=lambda x: x[1].get("overall_score", 0.0)
        )
        worst_trajectory, worst_score = min(
            scored_this_round, key=lambda x: x[1].get("overall_score", 0.0)
        )

        print(
            f"[grpo] prompt={prompt_id} best=attempt{best_trajectory['attempt_id']}({best_score.get('overall_score', 0.0)}) "
            f"vs worst=attempt{worst_trajectory['attempt_id']}({worst_score.get('overall_score', 0.0)})"
        )

        # GRPO Step 3: Compare best vs worst and write ONE experience to memory
        if len(scored_this_round) >= 2:
            retry_delay = max(0.0, attempt_retry_delay_s)
            grpo_success = False
            for retry_idx in range(max(0, attempt_retries) + 1):
                try:
                    print(f"[grpo] start prompt={prompt_id}")
                    await maybe_wait_for(
                        grpo_summarize_and_update_memory(
                            agent=judge_agent,
                            prompt=pending_for_prompt[0]["prompt"],
                            best_trajectory=best_trajectory,
                            worst_trajectory=worst_trajectory,
                            request_timeout_s=attempt_timeout_s,
                        ),
                        attempt_timeout_s,
                    )
                    grpo_success = True
                    print(f"[grpo] done prompt={prompt_id}")
                    break
                except asyncio.TimeoutError:
                    print(
                        f"[grpo] timeout prompt={prompt_id} after {attempt_timeout_s}s"
                    )
                except Exception as e:
                    print(f"[grpo] failed prompt={prompt_id} retry={retry_idx}: {e}")
                    if is_stream_error(e):
                        from tools.SubAgentTool import reset_cached_mcp_tools

                        reset_cached_mcp_tools()

                if retry_idx < max(0, attempt_retries):
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
                    retry_delay = (
                        retry_delay * attempt_retry_backoff
                        if attempt_retry_backoff > 0
                        else retry_delay
                    )

            if not grpo_success:
                print(
                    f"[grpo] skipped prompt={prompt_id} after {attempt_retries} retries"
                )
        else:
            print(
                f"[grpo] only {len(scored_this_round)} attempt(s), need >=2, skipping experience extraction"
            )

        # GRPO Step 4: Save all score records
        for trajectory, score in scored_this_round:
            key = (trajectory["prompt_id"], trajectory["attempt_id"])
            if key in scored_keys:
                continue
            score_record: Dict[str, Any] = {
                "prompt_id": trajectory["prompt_id"],
                "attempt_id": trajectory["attempt_id"],
                "prompt": trajectory["prompt"],
                "messages": trajectory["messages"],
                "response": trajectory["response"],
                "score": score,
                "meta": {
                    "created_at": int(time.time()),
                    "method": "grpo_judge_agent",
                },
            }
            append_jsonl(score_path, score_record)
            scored_keys.add(key)
            print(
                f"[score] saved prompt={trajectory['prompt_id']} attempt={trajectory['attempt_id']}"
            )

        # Reload memory so subsequent prompts can see the newly learned experiences
        memory_bank = load_memory_bank(memory_json_path)

        # Persist updated memory as markdown
        with open(memory_md_path, "w", encoding="utf-8") as f:
            f.write(render_memory_markdown(memory_bank))
        print(
            f"[memory] updated after prompt={prompt_id}; total={len(memory_bank['experiences'])}"
        )


async def build_agent(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    max_retries: int,
    backoff_factor: float,
    initial_delay: float,
    request_timeout_s: Optional[float],
    experiences: List[Dict[str, Any]] | None = None,
) -> Any:
    from deepagents import create_deep_agent
    from langchain.agents.middleware import ToolRetryMiddleware
    from langchain_openai import ChatOpenAI
    import tools.GeneralTool as GeneralToolModule
    from tools.SubAgentTool import init_subagents

    general_tools = []
    for func_name in GeneralToolModule.__all__:
        general_tools.append(getattr(GeneralToolModule, func_name))

    subagents = list(await init_subagents(experiences=experiences or []))

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        request_timeout=request_timeout_s,
    )

    return create_deep_agent(
        model=chat,
        tools=general_tools,
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


with open(
    "/Volumes/Samsung/Projects/robotagent/config/config.yml", "r", encoding="utf-8"
) as f:
    config = yaml.load(f.read(), yaml.FullLoader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect trajectories with online experience memory: "
            "collect -> score -> summarize-one -> append memory -> next round."
        )
    )
    parser.add_argument("--base_url", type=str, default=config["model_url"])
    parser.add_argument("--model", type=str, default=config["llm"])
    parser.add_argument("--api_key", type=str, default=config["api_key"])
    parser.add_argument("--prompts", type=str, default="SFT/data.txt")
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/training_free_grpo/trajectories.jsonl",
    )
    parser.add_argument(
        "--mirror_output_path",
        type=str,
        default="trajectories.jsonl",
        help="Optional secondary trajectories.jsonl path for downstream tooling.",
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
    parser.add_argument("--judge_model", type=str, default="deepseek-chat")
    parser.add_argument(
        "--judge_api_base", type=str, default="https://api.deepseek.com"
    )
    parser.add_argument("--judge_api_key", type=str, default="")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument(
        "--judge_max_tokens", type=int, default=8192
    )  # DeepSeek max output
    parser.add_argument("--max_experiences_in_prompt", type=int, default=20)
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-request timeout in seconds for model calls. Set <=0 to disable.",
    )
    parser.add_argument(
        "--attempt_timeout_s",
        type=float,
        default=DEFAULT_ATTEMPT_TIMEOUT_S,
        help="Overall timeout in seconds for each collect attempt. Set <=0 to disable.",
    )
    parser.add_argument(
        "--cleanup_timeout_s",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_S,
        help="Timeout in seconds for cleanup per attempt. Set <=0 to disable.",
    )
    parser.add_argument(
        "--attempt_retries",
        type=int,
        default=2,
        help="Retries for a collect attempt when stream errors occur.",
    )
    parser.add_argument(
        "--attempt_retry_delay_s",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds for attempt retries.",
    )
    parser.add_argument(
        "--attempt_retry_backoff",
        type=float,
        default=2.0,
        help="Backoff multiplier for attempt retry delay.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    request_timeout_s = normalize_timeout(args.request_timeout_s)
    attempt_timeout_s = normalize_timeout(args.attempt_timeout_s)
    cleanup_timeout_s = normalize_timeout(args.cleanup_timeout_s)
    config_path = os.path.join(root_dir, "config", "config.yml")
    mcp_cleanup_url = None
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
        mcp_cfg = cfg.get("mcp") or {}
        base = str(mcp_cfg.get("ip") or "http://localhost").rstrip("/")
        port = str(mcp_cfg.get("port") or "8001")
        mcp_cleanup_url = f"{base}:{port}/mcp"

        # Load judge config for DeepSeek defaults
        judge_cfg = cfg.get("judge") or {}
        if not args.judge_api_base or args.judge_api_base == "https://api.deepseek.com":
            args.judge_api_base = judge_cfg.get("api_base", "https://api.deepseek.com")
        if not args.judge_api_key:
            args.judge_api_key = judge_cfg.get("api_key", "")
        if not args.judge_model or args.judge_model == "deepseek-chat":
            args.judge_model = judge_cfg.get("model", "deepseek-chat")
    except Exception:
        mcp_cleanup_url = "http://localhost:8001/mcp"

    for path in [
        os.path.dirname(args.output_path),
        os.path.dirname(args.mirror_output_path),
        os.path.dirname(args.score_path),
        os.path.dirname(args.memory_json_path),
        os.path.dirname(args.memory_md_path),
    ]:
        if path:
            ensure_dir(path)

    prompts = load_prompts(args.prompts, args.max_prompts)
    base_system_prompt = resolve_base_system_prompt(args.system_prompt)

    async def agent_builder(
        system_prompt: str,
        experiences: list | None = None,
    ) -> Any:
        return await build_agent(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            system_prompt=system_prompt,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
            initial_delay=args.initial_delay,
            request_timeout_s=request_timeout_s,
            experiences=experiences,
        )

    judge_model_name = args.judge_model.strip() or "deepseek-chat"
    from langchain.agents.middleware import ToolRetryMiddleware
    from langchain_openai import ChatOpenAI

    judge_chat = ChatOpenAI(
        base_url=args.judge_api_base,
        model=judge_model_name,
        api_key=args.judge_api_key,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        request_timeout=request_timeout_s,
    )
    judge_agent = await build_judge_agent(
        model=judge_chat,
        tools=JUDGE_EXPERIENCE_TOOLS,
        middleware=[
            ToolRetryMiddleware(
                max_retries=args.max_retries,
                backoff_factor=args.backoff_factor,
                initial_delay=args.initial_delay,
            )
        ],
    )

    cleanup_hook = None
    init_hook = None
    if CLEANUP_SIMULATION_PER_ATTEMPT:
        from fastmcp import Client

        mcp_client = Client(mcp_cleanup_url)

        async def cleanup_hook() -> None:
            await mcp_client.call_tool("cleanup_simulation_tool")

        async def init_hook() -> None:
            await mcp_client.call_tool("initialize_simulation", {"args": {}})

        async with mcp_client:
            await collect_trajectories_online(
                agent_builder=agent_builder,
                prompts=prompts,
                output_path=args.output_path,
                mirror_output_path=args.mirror_output_path,
                score_path=args.score_path,
                memory_json_path=args.memory_json_path,
                memory_md_path=args.memory_md_path,
                samples_per_prompt=args.samples_per_prompt,
                rebuild_agent_every=REBUILD_AGENT_EVERY,
                base_system_prompt=base_system_prompt,
                judge_agent=judge_agent,
                judge_model=judge_chat,
                max_experiences_in_prompt=args.max_experiences_in_prompt,
                attempt_timeout_s=attempt_timeout_s,
                cleanup_timeout_s=cleanup_timeout_s,
                attempt_retries=args.attempt_retries,
                attempt_retry_delay_s=args.attempt_retry_delay_s,
                attempt_retry_backoff=args.attempt_retry_backoff,
                cleanup_after_attempt=cleanup_hook,
                init_before_attempt=init_hook,
            )
    else:
        await collect_trajectories_online(
            agent_builder=agent_builder,
            prompts=prompts,
            output_path=args.output_path,
            mirror_output_path=args.mirror_output_path,
            score_path=args.score_path,
            memory_json_path=args.memory_json_path,
            memory_md_path=args.memory_md_path,
            samples_per_prompt=args.samples_per_prompt,
            rebuild_agent_every=REBUILD_AGENT_EVERY,
            base_system_prompt=base_system_prompt,
            judge_agent=judge_agent,
            judge_model=judge_chat,
            max_experiences_in_prompt=args.max_experiences_in_prompt,
            attempt_timeout_s=attempt_timeout_s,
            cleanup_timeout_s=cleanup_timeout_s,
            attempt_retries=args.attempt_retries,
            attempt_retry_delay_s=args.attempt_retry_delay_s,
            attempt_retry_backoff=args.attempt_retry_backoff,
            cleanup_after_attempt=cleanup_hook,
            init_before_attempt=init_hook,
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
