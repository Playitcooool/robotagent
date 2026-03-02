import ast
import hashlib
import json
import re


def normalize_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def extract_text_from_message(msg) -> str:
    content = None
    content_blocks = None
    if isinstance(msg, dict):
        content = msg.get("content") or msg.get("text")
        content_blocks = msg.get("content_blocks")
    else:
        content = getattr(msg, "content", None) or getattr(msg, "text", None)
        content_blocks = getattr(msg, "content_blocks", None)

    text = normalize_text(content)
    if text:
        return text

    if isinstance(content_blocks, list):
        parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return "".join(parts)
    return ""


def normalize_message_role(msg) -> str:
    role = None
    if isinstance(msg, dict):
        role = msg.get("role")
    else:
        role = getattr(msg, "role", None)

    if isinstance(role, str):
        lowered = role.lower()
        if lowered in {"assistant", "ai"}:
            return "assistant"
        if lowered in {"user", "human"}:
            return "user"
        if lowered in {"tool", "function"}:
            return "tool"
        return lowered

    cls_name = msg.__class__.__name__
    if cls_name in {"AIMessage", "AIMessageChunk"}:
        return "assistant"
    if cls_name == "HumanMessage":
        return "user"
    if cls_name in {"ToolMessage", "ToolMessageChunk"}:
        return "tool"
    return "unknown"


def extract_message_name(msg) -> str:
    if isinstance(msg, dict):
        name = msg.get("name")
    else:
        name = getattr(msg, "name", None)
    return str(name or "").strip().lower()


def normalize_todo_status(raw_status: str) -> str:
    s = (raw_status or "").strip().lower()
    if s in {"completed", "done", "finished", "success"}:
        return "completed"
    if s in {"in_progress", "in progress", "running", "active", "doing"}:
        return "in_progress"
    return "pending"


def extract_todo_list_from_text(text: str):
    if not text:
        return []

    candidates = []
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        candidates.append(bracket_match.group(0))
    candidates.append(text)

    for raw in candidates:
        candidate = (raw or "").strip()
        if not candidate:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                continue
    return []


def extract_planning_steps_from_write_todos(content):
    text = normalize_text(content).strip()
    todo_items = extract_todo_list_from_text(text)
    if not isinstance(todo_items, list) or not todo_items:
        return []

    steps = []
    for idx, item in enumerate(todo_items, start=1):
        if not isinstance(item, dict):
            continue
        step_text = (
            item.get("content")
            or item.get("step")
            or item.get("title")
            or item.get("task")
            or ""
        )
        step_text = str(step_text).strip()
        if not step_text:
            continue
        steps.append(
            {
                "id": str(item.get("id") or idx),
                "step": step_text,
                "status": normalize_todo_status(str(item.get("status") or "")),
            }
        )
    return steps


def truncate_text(text: str, max_len: int = 200) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def safe_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def normalize_usage_payload(raw_usage) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return {}

    prompt_tokens = safe_int(
        raw_usage.get("prompt_tokens")
        or raw_usage.get("input_tokens")
        or raw_usage.get("prompt_token_count")
    )
    completion_tokens = safe_int(
        raw_usage.get("completion_tokens")
        or raw_usage.get("output_tokens")
        or raw_usage.get("completion_token_count")
    )
    total_tokens = safe_int(raw_usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return {}

    return {
        "prompt_tokens": max(prompt_tokens, 0),
        "completion_tokens": max(completion_tokens, 0),
        "total_tokens": max(total_tokens, 0),
    }


def extract_message_token_usage(msg) -> dict[str, int]:
    usage_candidates = []
    if isinstance(msg, dict):
        usage_candidates.append(msg.get("usage_metadata"))
        usage_candidates.append(msg.get("token_usage"))
        usage_candidates.append(msg.get("usage"))
        response_meta = msg.get("response_metadata")
        if isinstance(response_meta, dict):
            usage_candidates.append(response_meta.get("token_usage"))
            usage_candidates.append(response_meta.get("usage"))
            usage_candidates.append(response_meta.get("usage_metadata"))
    else:
        usage_candidates.append(getattr(msg, "usage_metadata", None))
        usage_candidates.append(getattr(msg, "token_usage", None))
        response_meta = getattr(msg, "response_metadata", None)
        if isinstance(response_meta, dict):
            usage_candidates.append(response_meta.get("token_usage"))
            usage_candidates.append(response_meta.get("usage"))
            usage_candidates.append(response_meta.get("usage_metadata"))

    for raw in usage_candidates:
        normalized = normalize_usage_payload(raw)
        if normalized:
            return normalized
    return {}


def extract_message_id(msg) -> str:
    if isinstance(msg, dict):
        msg_id = msg.get("id")
        msg_type = msg.get("type") or msg.get("role")
        content = normalize_text(msg.get("content") or msg.get("text"))
    else:
        msg_id = getattr(msg, "id", None)
        msg_type = msg.__class__.__name__
        content = normalize_text(
            getattr(msg, "content", None) or getattr(msg, "text", None)
        )

    if msg_id is not None and str(msg_id).strip():
        return str(msg_id).strip()

    fingerprint = f"{msg_type}:{content[:120]}"
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def sum_usage_map(usage_by_message: dict[str, dict[str, int]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usage_by_message.values():
        prompt_tokens += safe_int(usage.get("prompt_tokens"))
        completion_tokens += safe_int(usage.get("completion_tokens"))
        total_tokens += safe_int(usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
