import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Set, Tuple

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
                finished.add((item["prompt_id"], item["attempt_id"]))
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


async def collect_trajectories(
    agent_builder: Callable[[], Awaitable[Any]],
    prompts: List[str],
    output_path: str,
    samples_per_prompt: int,
    rebuild_agent_every: int,
    cleanup_after_attempt: Callable[[], Awaitable[None]] | None = None,
) -> None:
    finished_keys = load_finished_keys(output_path)
    print(f"[collect] existing trajectories: {len(finished_keys)}")
    agent = None
    attempt_since_rebuild = 0

    for prompt_id, prompt in enumerate(prompts):
        for attempt_id in range(samples_per_prompt):
            key = (prompt_id, attempt_id)
            if key in finished_keys:
                continue

            try:
                # Rebuild agent periodically to prevent long-run resource growth.
                if (
                    agent is None
                    or rebuild_agent_every > 0
                    and attempt_since_rebuild >= rebuild_agent_every
                ):
                    agent = await agent_builder()
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

            record = {
                "prompt_id": prompt_id,
                "attempt_id": attempt_id,
                "prompt": prompt,
                "messages": trajectory_messages,
                "response": response,
                "meta": {
                    "method": "deep_agent",
                    "created_at": int(time.time()),
                },
            }
            append_jsonl(output_path, record)
            finished_keys.add(key)
            print(f"[collect] saved prompt={prompt_id} attempt={attempt_id}")


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
    from prompts import MainAgentPrompt
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

    prompt = system_prompt.strip() if system_prompt.strip() else MainAgentPrompt.SYSTEM_PROMPT

    return create_deep_agent(
        model=chat,
        tools=[],
        system_prompt=prompt,
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
        description="Collect trajectories with deep agent only."
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
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--samples_per_prompt", type=int, default=3)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    parser.add_argument("--initial_delay", type=float, default=1.0)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        ensure_dir(output_dir)

    prompts = load_prompts(args.prompts, args.max_prompts)

    async def agent_builder() -> Any:
        return await build_agent(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
            initial_delay=args.initial_delay,
        )

    cleanup_hook = None
    if CLEANUP_SIMULATION_PER_ATTEMPT:
        from fastmcp import Client

        mcp_client = Client(MCP_CLEANUP_URL)

        async def cleanup_hook() -> None:
            await mcp_client.call_tool("cleanup_simulation_tool")

        async with mcp_client:
            await collect_trajectories(
                agent_builder=agent_builder,
                prompts=prompts,
                output_path=args.output_path,
                samples_per_prompt=args.samples_per_prompt,
                rebuild_agent_every=REBUILD_AGENT_EVERY,
                cleanup_after_attempt=cleanup_hook,
            )
    else:
        await collect_trajectories(
            agent_builder=agent_builder,
            prompts=prompts,
            output_path=args.output_path,
            samples_per_prompt=args.samples_per_prompt,
            rebuild_agent_every=REBUILD_AGENT_EVERY,
            cleanup_after_attempt=cleanup_hook,
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
