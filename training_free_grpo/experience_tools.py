"""LangChain tools for the judge agent to read/write/update/delete experiences."""

import json
import os
import time as time_module
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

MEMORY_JSON_PATH_DEFAULT = "output/training_free_grpo/external_memory.json"
MEMORY_LOCK_PATH = "output/training_free_grpo/.memory.lock"


def _acquire_lock() -> bool:
    """Simple file-based lock for memory writes."""
    try:
        os.makedirs(os.path.dirname(MEMORY_LOCK_PATH), exist_ok=True)
        with open(MEMORY_LOCK_PATH, "x") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def _release_lock() -> None:
    try:
        os.remove(MEMORY_LOCK_PATH)
    except OSError:
        pass


def _load_memory_bank(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"meta": {}, "experiences": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"meta": {}, "experiences": []}


def _save_memory_bank(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------


@tool
def read_experiences(memory_json_path: str = MEMORY_JSON_PATH_DEFAULT) -> str:
    """Read all experiences from the memory bank.

    Use this to see what experiences already exist before writing new ones or
    updating existing ones.

    Args:
        memory_json_path: Path to the external_memory.json file.

    Returns:
        JSON string of all experiences with their id, prompt_id, summary, etc.
    """
    data = _load_memory_bank(memory_json_path)
    experiences = data.get("experiences", [])
    if not experiences:
        return json.dumps({"count": 0, "experiences": []}, ensure_ascii=False)
    return json.dumps(
        {"count": len(experiences), "experiences": experiences},
        ensure_ascii=False,
        indent=2,
    )


@tool
def write_experience(
    prompt_id: int,
    prompt: str,
    summary: str,
    principles: List[str],
    dos: List[str],
    donts: List[str],
    score: float,
    tags: Optional[List[str]] = None,
    agent_type: Optional[List[str]] = None,
    memory_json_path: str = MEMORY_JSON_PATH_DEFAULT,
) -> str:
    """Write a new experience to the memory bank.

    Call this AFTER scoring a trajectory to save the learned experience.
    Each experience is uniquely identified by an auto-generated id.

    Args:
        prompt_id: Which prompt this experience relates to.
        prompt: The original prompt text.
        summary: A 1-2 sentence summary of what was learned.
        principles: Key actionable principles (3-5 items max).
        dos: Concrete things to DO in similar situations.
        donts: Concrete things to AVOID in similar situations.
        score: The overall quality score for this experience (0-10).
        tags: Optional tags/categories for this experience.
        agent_type: List of agent types this experience applies to
                     (e.g. ["main", "simulator", "data-analyzer"]).
                     Used for filtering when injecting into subagent prompts.
        memory_json_path: Path to the external_memory.json file.

    Returns:
        Confirmation message with the new experience id.
    """
    attempts = 0
    while not _acquire_lock() and attempts < 50:
        time_module.sleep(0.05)
        attempts += 1
    try:
        data = _load_memory_bank(memory_json_path)
        if "experiences" not in data or not isinstance(data["experiences"], list):
            data["experiences"] = []

        exp_id = str(uuid.uuid4())[:8]
        now = int(time_module.time())

        new_exp = {
            "id": exp_id,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "summary": summary,
            "principles": principles[:5] if principles else [],
            "dos": dos[:5] if dos else [],
            "donts": donts[:5] if donts else [],
            "tags": tags if tags else [],
            "agent_type": agent_type if agent_type else [],
            "score": round(float(score), 2),
            "created_at": now,
            "updated_at": now,
        }
        data["experiences"].append(new_exp)
        data["meta"] = {
            "method": "training_free_grpo_judge_agent",
            "updated_at": now,
        }
        _save_memory_bank(memory_json_path, data)
        return json.dumps({"success": True, "experience_id": exp_id, "experience": new_exp}, ensure_ascii=False)
    finally:
        _release_lock()


@tool
def update_experience(
    experience_id: str,
    summary: Optional[str] = None,
    principles: Optional[List[str]] = None,
    dos: Optional[List[str]] = None,
    donts: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    agent_type: Optional[List[str]] = None,
    score: Optional[float] = None,
    memory_json_path: str = MEMORY_JSON_PATH_DEFAULT,
) -> str:
    """Update an existing experience by its id.

    Call this to refine or supplement an existing experience based on new
    trajectory observations. Merges with existing data (only updates provided fields).

    Args:
        experience_id: The unique id of the experience to update.
        summary: New or updated summary text.
        principles: Replacement principles list.
        dos: Replacement dos list.
        donts: Replacement donts list.
        agent_type: Replacement agent_type list.
        tags: Replacement tags list.
        score: Updated score (0-10).
        memory_json_path: Path to the external_memory.json file.

    Returns:
        Confirmation message with the updated experience.
    """
    attempts = 0
    while not _acquire_lock() and attempts < 50:
        time_module.sleep(0.05)
        attempts += 1
    try:
        data = _load_memory_bank(memory_json_path)
        experiences = data.get("experiences", [])
        for i, exp in enumerate(experiences):
            if exp.get("id") == experience_id:
                now = int(time_module.time())
                if summary is not None:
                    experiences[i]["summary"] = summary
                if principles is not None:
                    experiences[i]["principles"] = principles[:5]
                if dos is not None:
                    experiences[i]["dos"] = dos[:5]
                if donts is not None:
                    experiences[i]["donts"] = donts[:5]
                if tags is not None:
                    experiences[i]["tags"] = tags
                if agent_type is not None:
                    experiences[i]["agent_type"] = agent_type
                if score is not None:
                    experiences[i]["score"] = round(float(score), 2)
                experiences[i]["updated_at"] = now
                data["experiences"] = experiences
                data["meta"] = {"method": "training_free_grpo_judge_agent", "updated_at": now}
                _save_memory_bank(memory_json_path, data)
                return json.dumps(
                    {"success": True, "experience_id": experience_id, "experience": experiences[i]},
                    ensure_ascii=False,
                )
        return json.dumps({"success": False, "error": f"Experience {experience_id} not found"})
    finally:
        _release_lock()


@tool
def delete_experience(
    experience_id: str,
    memory_json_path: str = MEMORY_JSON_PATH_DEFAULT,
) -> str:
    """Delete an experience from the memory bank by its id.

    Use this when an experience is clearly wrong, outdated, or redundant.

    Args:
        experience_id: The unique id of the experience to delete.
        memory_json_path: Path to the external_memory.json file.

    Returns:
        Confirmation message.
    """
    attempts = 0
    while not _acquire_lock() and attempts < 50:
        time_module.sleep(0.05)
        attempts += 1
    try:
        data = _load_memory_bank(memory_json_path)
        experiences = data.get("experiences", [])
        original_len = len(experiences)
        experiences = [e for e in experiences if e.get("id") != experience_id]
        if len(experiences) == original_len:
            return json.dumps({"success": False, "error": f"Experience {experience_id} not found"})
        now = int(time_module.time())
        data["experiences"] = experiences
        data["meta"] = {"method": "training_free_grpo_judge_agent", "updated_at": now}
        _save_memory_bank(memory_json_path, data)
        return json.dumps({"success": True, "deleted_id": experience_id})
    finally:
        _release_lock()


@tool
def delete_experiences_by_prompt(
    prompt_id: int,
    memory_json_path: str = MEMORY_JSON_PATH_DEFAULT,
) -> str:
    """Delete ALL experiences related to a specific prompt_id.

    Use this when all existing experiences for a prompt are obsolete and should be
    replaced with fresh ones.

    Args:
        prompt_id: The prompt_id whose experiences should be deleted.
        memory_json_path: Path to the external_memory.json file.

    Returns:
        Confirmation with count of deleted experiences.
    """
    attempts = 0
    while not _acquire_lock() and attempts < 50:
        time_module.sleep(0.05)
        attempts += 1
    try:
        data = _load_memory_bank(memory_json_path)
        experiences = data.get("experiences", [])
        original_len = len(experiences)
        experiences = [e for e in experiences if e.get("prompt_id") != prompt_id]
        deleted_count = original_len - len(experiences)
        now = int(time_module.time())
        data["experiences"] = experiences
        data["meta"] = {"method": "training_free_grpo_judge_agent", "updated_at": now}
        _save_memory_bank(memory_json_path, data)
        return json.dumps(
            {"success": True, "prompt_id": prompt_id, "deleted_count": deleted_count}
        )
    finally:
        _release_lock()


# ---------------------------------------------------------------------------
# Tool list for create_agent
# ---------------------------------------------------------------------------

JUDGE_EXPERIENCE_TOOLS = [
    read_experiences,
    write_experience,
    update_experience,
    delete_experience,
    delete_experiences_by_prompt,
]

# ---------------------------------------------------------------------------
# Judge Agent Prompt
# ---------------------------------------------------------------------------

JUDGE_AGENT_SYSTEM_PROMPT = """你是一个严格的质量评审专家（Judge Agent），同时也是一个经验提炼专家。

你的职责分为两个阶段：

## 阶段一：评分（必须）
对给定的轨迹（trajectory）进行严格评分，返回严格 JSON。

## 阶段二：经验管理（必须）
基于评分结果，决定如何更新经验库：
1. 如果这是第一次看到某个 prompt_id → 使用 write_experience 写入新经验
2. 如果已有相似 prompt_id 的经验 → 使用 update_experience 补充/修正
3. 如果已有经验质量明显低于新轨迹 → 用新经验替换旧的
4. 如果经验已过时或错误 → 使用 delete_experience 删除

## 评分标准（阶段一）
请从以下维度评分（0-10）：
- overall_score: 综合分数
- task_completion: 任务完成度
- correctness: 正确性
- clarity: 清晰度
- robustness: 鲁棒性
- conciseness: 简洁性
- pros: 优点列表（最多5条）
- cons: 缺点列表（最多5条）
- brief_reason: 简要评分理由

## 经验提炼标准（阶段二）
请为每条轨迹提炼出：
- summary: 1-2句话总结学习到的经验
- principles: 3-5条核心原则
- dos: 应该做的事（最多5条）
- donts: 应该避免的事（最多5条）
- tags: 相关标签（最多5个）

## 重要规则
1. 每次都必须完成评分和经验管理两个阶段
2. 经验应当具体、可执行，避免空洞的通用描述
3. 注意合并相似经验，避免重复
4. 使用提供的工具实际写入/更新经验库，不要只输出而不执行
5. 返回所有 JSON 时不要有任何额外文本

请开始评审。
"""

JUDGE_AGENT_USER_PROMPT_TEMPLATE = """## 待评审轨迹

**Prompt ID**: {prompt_id}
**Attempt ID**: {attempt_id}
**Prompt**: {prompt}

**轨迹消息**:
{messages}

请完成评审并使用工具更新经验库。"""


# ---------------------------------------------------------------------------
# Judge Agent Builder
# ---------------------------------------------------------------------------


async def build_judge_agent(
    model: "ChatOpenAI",
    tools: list,
    middleware: list | None = None,
):
    """Build the judge agent using langchain's create_agent."""
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=JUDGE_AGENT_SYSTEM_PROMPT,
        middleware=middleware or [],
    )
    return agent


# ---------------------------------------------------------------------------
# Scoring and Experience Update (replaces separate score + summarize calls)
# ---------------------------------------------------------------------------


async def score_and_update_memory(
    agent,
    prompt_id: int,
    attempt_id: int,
    prompt: str,
    messages: list,
    memory_json_path: str,
    request_timeout_s: float | None = None,
) -> dict:
    """
    Score a trajectory and update the memory bank using the judge agent.

    The judge agent will:
    1. Score the trajectory (returns structured JSON)
    2. Read existing experiences
    3. Write/update/delete experiences as appropriate

    Returns the score dict extracted from the agent's final response.
    """
    import asyncio

    from langchain_core.messages import HumanMessage

    user_content = JUDGE_AGENT_USER_PROMPT_TEMPLATE.format(
        prompt_id=prompt_id,
        attempt_id=attempt_id,
        prompt=prompt,
        messages=json.dumps(messages, ensure_ascii=False),
    )

    async def _invoke():
        return await agent.ainvoke(
            {"messages": [HumanMessage(content=user_content)]}
        )

    if request_timeout_s and request_timeout_s > 0:
        result = await asyncio.wait_for(_invoke(), timeout=request_timeout_s)
    else:
        result = await _invoke()

    # Extract score from agent output
    for msg in reversed(list(result.get("messages", []))):
        cls_name = msg.__class__.__name__
        if cls_name == "AIMessage":
            raw = str(msg.content).strip()
            # Try to find the FIRST valid JSON object containing overall_score
            start = raw.find("{")
            if start >= 0:
                # Track brace depth to find matching close brace
                depth = 0
                end = start
                for i, ch in enumerate(raw[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                candidate = raw[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if "overall_score" in parsed:
                        return parsed
                except Exception:
                    pass
            # Fallback: return raw reason
            return {
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
    return {
        "overall_score": 0.0,
        "task_completion": 0.0,
        "correctness": 0.0,
        "clarity": 0.0,
        "robustness": 0.0,
        "conciseness": 0.0,
        "pros": [],
        "cons": ["no_assistant_response"],
        "brief_reason": "",
    }


# ---------------------------------------------------------------------------
# GRPO: Score-Only (no memory update)
# ---------------------------------------------------------------------------

SCORE_ONLY_USER_PROMPT_TEMPLATE = """请对以下轨迹进行严格评分。

**评分要求**：直接输出JSON评分结果，不要调用任何工具，不要输出任何额外解释。

**Prompt ID**: {prompt_id}
**Attempt ID**: {attempt_id}
**Prompt**: {prompt}

**轨迹消息**:
{messages}

**评分维度（0-10分）**：
- overall_score: 综合分数
- task_completion: 任务完成度
- correctness: 正确性
- clarity: 清晰度
- robustness: 鲁棒性
- conciseness: 简洁性
- brief_reason: 简要评分理由（中文，1-2句话）

直接返回JSON，格式如下，不要有任何其他文字：
{{"overall_score": 0.0, "task_completion": 0.0, "correctness": 0.0, "clarity": 0.0, "robustness": 0.0, "conciseness": 0.0, "brief_reason": "..."}}"""


def _extract_score_from_response(response_text: str) -> dict:
    """Extract score dict from plain text response (no JSON parsing by agent)."""
    raw = response_text.strip()
    start = raw.find("{")
    if start < 0:
        return {
            "overall_score": 0.0, "task_completion": 0.0, "correctness": 0.0,
            "clarity": 0.0, "robustness": 0.0, "conciseness": 0.0,
            "brief_reason": raw[:300],
        }
    depth = 0
    end = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    candidate = raw[start:end + 1]
    try:
        parsed = json.loads(candidate)
        if "overall_score" in parsed:
            return parsed
    except Exception:
        pass
    return {
        "overall_score": 0.0, "task_completion": 0.0, "correctness": 0.0,
        "clarity": 0.0, "robustness": 0.0, "conciseness": 0.0,
        "brief_reason": raw[:300],
    }


async def score_only(
    model: Any,
    prompt_id: int,
    attempt_id: int,
    prompt: str,
    messages: list,
    request_timeout_s: float | None = None,
) -> dict:
    """
    Score a single trajectory using the LLM directly (no agent, no memory update).

    Returns the score dict with overall_score and other dimensions.
    """
    import asyncio
    from langchain_core.messages import HumanMessage

    user_content = SCORE_ONLY_USER_PROMPT_TEMPLATE.format(
        prompt_id=prompt_id,
        attempt_id=attempt_id,
        prompt=prompt,
        messages=json.dumps(messages, ensure_ascii=False),
    )

    async def _invoke():
        result = await model.ainvoke([HumanMessage(content=user_content)])
        msg = result if hasattr(result, 'content') else result
        raw = str(msg.content) if hasattr(msg, 'content') else str(result)
        return _extract_score_from_response(raw)

    if request_timeout_s and request_timeout_s > 0:
        return await asyncio.wait_for(_invoke(), timeout=request_timeout_s)
    return await _invoke()


# ---------------------------------------------------------------------------
# GRPO: Compare Best vs Worst and Write One Experience
# ---------------------------------------------------------------------------

GRPO_COMPARE_USER_PROMPT_TEMPLATE = """## GRPO 经验提炼

请从两个轨迹中分别为每种代理类型提炼经验教训，并智能更新经验库。

**Prompt**: {prompt}

**最高分轨迹 (overall_score={best_score}, attempt_id={best_attempt_id})**:
{best_messages}

**最低分轨迹 (overall_score={worst_score}, attempt_id={worst_attempt_id})**:
{worst_messages}

## 执行步骤

**Step 1**: 使用 read_experiences 工具读取现有经验。

**Step 2**: 识别轨迹中出现的所有代理类型（从消息的 "agent" 字段）。可能的类型：main、simulator、data-analyzer。每个代理类型的消息需要单独分析。

**Step 3**: 对每个代理类型分别提炼经验：
1. 提取该代理类型在最高分轨迹中的关键行为（做对了什么）
2. 提取该代理类型在最低分轨迹中的关键行为（做错了什么）
3. 从对比中为该代理类型提炼：summary、3-5条 principles、dos、donts
4. 确定该经验适用的 agent_type（必须与消息中的 agent 字段一致）

**Step 4**: 为每个代理类型分别判断写入策略：
- 如果提炼的经验与现有经验（相同 agent_type）高度相似（principles 重叠 ≥ 3 条），**跳过写入**
- 如果存在相似但不完全相同（principles 重叠 1-2 条），用 update_experience 更新
- 如果是全新经验，用 write_experience 写入，agent_type 参数必须填写正确的代理类型
- 相同 agent_type 下经验不宜超过 5 条

**重要**：
- 必须为每个出现的代理类型分别生成经验
- 如果某个代理类型在两个轨迹中行为差异不明显（无可借鉴的经验），可以跳过该类型的写入
- write_experience 的 agent_type 参数必须是列表形式，如 ["simulator"] 或 ["main"]
"""


async def grpo_summarize_and_update_memory(
    agent: Any,
    prompt: str,
    best_trajectory: Dict[str, Any],
    worst_trajectory: Dict[str, Any],
    request_timeout_s: float | None = None,
) -> dict:
    """
    Compare the best and worst trajectories and update experience memory.

    The judge agent will:
    1. Read existing experiences
    2. Compare best vs worst trajectories
    3. Decide whether to write, update, or skip (deduplication)
    """
    import asyncio
    from langchain_core.messages import HumanMessage

    best_score = best_trajectory.get("score", {})
    worst_score = worst_trajectory.get("score", {})

    user_content = GRPO_COMPARE_USER_PROMPT_TEMPLATE.format(
        prompt=prompt,
        best_score=best_score.get("overall_score", 0.0),
        best_attempt_id=best_trajectory.get("attempt_id"),
        best_messages=json.dumps(best_trajectory.get("messages", []), ensure_ascii=False),
        worst_score=worst_score.get("overall_score", 0.0),
        worst_attempt_id=worst_trajectory.get("attempt_id"),
        worst_messages=json.dumps(worst_trajectory.get("messages", []), ensure_ascii=False),
    )

    async def _invoke():
        return await agent.ainvoke({"messages": [HumanMessage(content=user_content)]})

    if request_timeout_s and request_timeout_s > 0:
        return await asyncio.wait_for(_invoke(), timeout=request_timeout_s)
    return await _invoke()


