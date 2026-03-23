import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,host.docker.internal")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,host.docker.internal")

from fastapi import FastAPI
from fastapi import Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
import ast
from deepagents import create_deep_agent
from tools.SubAgentTool import init_subagents, _load_agent_experiences, build_experience_suffix
import logging
import json
import yaml
from tools import GeneralTool
from tools import AnalysisTool
import time
from langchain_openai import ChatOpenAI
import os
import re
import hashlib
from prompts import MainAgentPrompt
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redis.asyncio import Redis
from typing import Optional
from backend.schemas import ChatIn
from backend.auth_utils import (
    validate_username,
    validate_password,
    hash_password,
    verify_password,
)
from backend.stream_utils import (
    normalize_text,
    extract_text_from_message,
    extract_thinking_from_message,
    split_think_and_answer_delta,
    normalize_message_role,
    extract_message_name,
    extract_planning_steps_from_write_todos,
    truncate_text,
    safe_int,
    extract_message_token_usage,
    extract_message_id,
    sum_usage_map,
)
from backend.routes_auth import register_auth_routes
from backend.routes_sim import register_sim_routes

os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
# Chat history Redis should be separated from agent checkpoint Redis.
os.environ.setdefault("CHAT_REDIS_URL", "redis://127.0.0.1:6379/1")
# Auth/user Redis should be separated from chat history Redis.
os.environ.setdefault("AUTH_REDIS_URL", "redis://127.0.0.1:6379/2")
# Shared realtime frame location written by mcp/mcp_server.py
# Default path points to repo-mounted directory so host and docker can share files.
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SIM_STREAM_DIR = (ROOT_DIR / "mcp" / ".sim_stream").resolve()
SIM_STREAM_DIR = Path(
    os.environ.get("PYBULLET_STREAM_DIR", str(DEFAULT_SIM_STREAM_DIR))
).resolve()
SIM_META_FILE = SIM_STREAM_DIR / "latest.json"
SIM_FRAME_FILE = SIM_STREAM_DIR / "latest.png"
_SIM_FRAME_CACHE = {
    "meta_mtime": 0.0,
    "frame_mtime": 0.0,
    "payload": {"status": "idle", "has_frame": False},
}
# ========== 1. 日志配置（确保输出到Uvicorn控制台） ==========
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
DEBUG_STREAM_FIELDS = os.environ.get("DEBUG_STREAM_FIELDS", "0") == "1"

# ========== 2. 全局变量定义（关键：提前声明active_agent） ==========
active_agent = None  # 全局agent，启动事件中初始化
chat_redis: Optional[Redis] = None
auth_redis: Optional[Redis] = None
CHAT_REDIS_URL = os.environ["CHAT_REDIS_URL"]
AUTH_REDIS_URL = os.environ["AUTH_REDIS_URL"]
CHAT_HISTORY_PREFIX = "robotagent:chat:messages"
CHAT_HISTORY_MAX_LEN = int(os.environ.get("CHAT_HISTORY_MAX_LEN", "200"))
CHAT_SESSIONS_ZSET_PREFIX = "robotagent:chat:sessions"
AUTH_USER_PREFIX = "robotagent:auth:user"
AUTH_SESSION_PREFIX = "robotagent:auth:session"
AUTH_SESSION_TTL_SECONDS = int(
    os.environ.get("AUTH_SESSION_TTL_SECONDS", str(30 * 24 * 3600))
)
PASSWORD_PBKDF2_ITERATIONS = int(
    os.environ.get("AUTH_PASSWORD_PBKDF2_ITERATIONS", "200000")
)
with open(ROOT_DIR / "config" / "config.yml", "r", encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)


# ========== 4. 加载工具函数（保留并添加日志） ==========
def get_tools():
    try:
        general_tools = []
        subagent_tools = []
        for func_name in GeneralTool.__all__:
            function = getattr(GeneralTool, func_name)
            general_tools.append(function)

        # 合并工具
        all_tools = general_tools + subagent_tools
        logger.info(f"总工具数量：{len(all_tools)}")
        return all_tools
    except Exception as e:
        logger.error(f"加载工具失败：{str(e)}", exc_info=True)
        return []  # 兜底返回空列表


# ========== 5. 初始化LLM模型 ==========
chatBot = ChatOpenAI(
    base_url=config["model_url"],
    model=config["llm"],
    api_key=config.get("api_key", "no_need"),
    streaming=True,
)

# ========== 6. FastAPI应用初始化 ==========
app = FastAPI()


# ========== 7. 启动事件（正确初始化agent） ==========
@app.on_event("startup")
async def startup_event():
    global active_agent, chat_redis, auth_redis  # 关联全局变量
    try:
        chat_redis = Redis.from_url(CHAT_REDIS_URL, decode_responses=True)
        await chat_redis.ping()
        logger.info(f"Chat history Redis 已连接: {CHAT_REDIS_URL}")
        auth_redis = Redis.from_url(AUTH_REDIS_URL, decode_responses=True)
        await auth_redis.ping()
        logger.info(f"Auth Redis 已连接: {AUTH_REDIS_URL}")

        # 加载所有工具
        all_tools = get_tools()
        if not all_tools:
            logger.warning("未加载到任何工具，agent将使用空工具列表")

        # 加载 experience（从 agent_experperiences.json）
        # 通过环境变量控制：WITHOUT_EXPERIENCE=1 启动则跳过注入
        inject_experiences = os.environ.get("WITHOUT_EXPERIENCE", "0") != "1"
        if inject_experiences:
            experiences = _load_agent_experiences()
            logger.info(f"已加载 {len(experiences)} 条 agent experiences")
        else:
            experiences = []
            logger.info("WITHOUT_EXPERIENCE=1，已跳过 experience 注入")

        subagents = list(await init_subagents(experiences=experiences))

        def _subagent_name(sa) -> str:
            if isinstance(sa, dict):
                return str(sa.get("name") or "").strip()
            return str(getattr(sa, "name", "") or "").strip()

        available_subagents = [_subagent_name(sa) for sa in subagents]
        available_subagents = [n for n in available_subagents if n]
        logger.info(f"可用子代理：{available_subagents}")

        prompt_suffix = (
            "\n\n[Runtime Subagent Availability]\n"
            f"- Available subagents now: {available_subagents or ['none']}\n"
        )
        if "simulator" not in available_subagents:
            prompt_suffix += (
                "- simulator is currently unavailable. "
                "Do NOT invoke simulator. "
                "For simulation requests, first explain simulator is unavailable and ask user to start MCP services.\n"
            )
        if "data-analyzer" not in available_subagents:
            prompt_suffix += (
                "- data-analyzer is currently unavailable. "
                "Do NOT invoke data-analyzer.\n"
            )

        # 注入 experience 到 MainAgent（simulation 和 analysis subagent 已通过 init_subagents 注入）
        if experiences:
            exp_suffix = build_experience_suffix(experiences)
        else:
            exp_suffix = ""

        runtime_system_prompt = MainAgentPrompt.SYSTEM_PROMPT + exp_suffix + prompt_suffix

        # 创建带工具的agent（核心修正）
        DB_URI = "redis://localhost:6379"
        async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.asetup()
            active_agent = create_deep_agent(
                model=chatBot,
                tools=all_tools,
                system_prompt=runtime_system_prompt,
                subagents=subagents,
                checkpointer=checkpointer,
            )
        logger.info("Agent 初始化成功！")
    except Exception as e:
        logger.error(f"Agent 初始化失败：{str(e)}", exc_info=True)
        active_agent = None
        if chat_redis is not None:
            await chat_redis.aclose()
            chat_redis = None
        if auth_redis is not None:
            await auth_redis.aclose()
            auth_redis = None


@app.on_event("shutdown")
async def shutdown_event():
    global chat_redis, auth_redis
    if chat_redis is not None:
        await chat_redis.aclose()
        chat_redis = None
    if auth_redis is not None:
        await auth_redis.aclose()
        auth_redis = None


# ========== 8. CORS中间件 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chat_history_key(user_id: str, session_id: str) -> str:
    safe = (session_id or "default_session").strip() or "default_session"
    return f"{CHAT_HISTORY_PREFIX}:{user_id}:{safe}"


def _chat_sessions_key(user_id: str) -> str:
    return f"{CHAT_SESSIONS_ZSET_PREFIX}:{user_id}"


def _auth_user_key(username: str) -> str:
    return f"{AUTH_USER_PREFIX}:{username.lower()}"


def _auth_session_key(token: str) -> str:
    return f"{AUTH_SESSION_PREFIX}:{token}"


async def _append_chat_message(user_id: str, session_id: str, role: str, text: str):
    if chat_redis is None:
        return
    now_ts = time.time()
    payload = {
        "id": int(now_ts * 1000),
        "role": role,
        "text": text or "",
        "session_id": session_id,
        "created_at": now_ts,
    }
    key = _chat_history_key(user_id, session_id)
    await chat_redis.rpush(key, json.dumps(payload, ensure_ascii=False))
    await chat_redis.ltrim(key, -CHAT_HISTORY_MAX_LEN, -1)
    await chat_redis.zadd(_chat_sessions_key(user_id), {session_id: now_ts})


def _normalize_text(content) -> str:
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


def _extract_text_from_message(msg) -> str:
    """Extract plain text from message/message-chunk structures."""
    content = None
    content_blocks = None
    if isinstance(msg, dict):
        content = msg.get("content") or msg.get("text")
        content_blocks = msg.get("content_blocks")
    else:
        content = getattr(msg, "content", None) or getattr(msg, "text", None)
        content_blocks = getattr(msg, "content_blocks", None)

    text = _normalize_text(content)
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


def _normalize_message_role(msg) -> str:
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


def _extract_message_name(msg) -> str:
    if isinstance(msg, dict):
        name = msg.get("name")
    else:
        name = getattr(msg, "name", None)
    return str(name or "").strip().lower()


def _normalize_todo_status(raw_status: str) -> str:
    s = (raw_status or "").strip().lower()
    if s in {"completed", "done", "finished", "success"}:
        return "completed"
    if s in {"in_progress", "in progress", "running", "active", "doing"}:
        return "in_progress"
    return "pending"


def _extract_todo_list_from_text(text: str):
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


def _extract_planning_steps_from_write_todos(content):
    text = _normalize_text(content).strip()
    todo_items = _extract_todo_list_from_text(text)
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
                "status": _normalize_todo_status(str(item.get("status") or "")),
            }
        )
    return steps


def _truncate_text(text: str, max_len: int = 200) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def _safe_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _normalize_usage_payload(raw_usage) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return {}

    prompt_tokens = _safe_int(
        raw_usage.get("prompt_tokens")
        or raw_usage.get("input_tokens")
        or raw_usage.get("prompt_token_count")
    )
    completion_tokens = _safe_int(
        raw_usage.get("completion_tokens")
        or raw_usage.get("output_tokens")
        or raw_usage.get("completion_token_count")
    )
    total_tokens = _safe_int(raw_usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return {}

    return {
        "prompt_tokens": max(prompt_tokens, 0),
        "completion_tokens": max(completion_tokens, 0),
        "total_tokens": max(total_tokens, 0),
    }


def _extract_message_token_usage(msg) -> dict[str, int]:
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
        normalized = _normalize_usage_payload(raw)
        if normalized:
            return normalized
    return {}


def _extract_message_id(msg) -> str:
    if isinstance(msg, dict):
        msg_id = msg.get("id")
        msg_type = msg.get("type") or msg.get("role")
        content = _normalize_text(msg.get("content") or msg.get("text"))
    else:
        msg_id = getattr(msg, "id", None)
        msg_type = msg.__class__.__name__
        content = _normalize_text(
            getattr(msg, "content", None) or getattr(msg, "text", None)
        )

    if msg_id is not None and str(msg_id).strip():
        return str(msg_id).strip()

    fingerprint = f"{msg_type}:{content[:120]}"
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _sum_usage_map(usage_by_message: dict[str, dict[str, int]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for usage in usage_by_message.values():
        prompt_tokens += _safe_int(usage.get("prompt_tokens"))
        completion_tokens += _safe_int(usage.get("completion_tokens"))
        total_tokens += _safe_int(usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


# bind extracted module helpers so endpoint logic uses modularized implementations
_validate_username = validate_username
_validate_password = validate_password
_hash_password = hash_password
_verify_password = verify_password
_normalize_text = normalize_text
_extract_text_from_message = extract_text_from_message
_extract_thinking_from_message = extract_thinking_from_message
_split_think_and_answer_delta = split_think_and_answer_delta
_normalize_message_role = normalize_message_role
_extract_message_name = extract_message_name
_extract_planning_steps_from_write_todos = extract_planning_steps_from_write_todos
_truncate_text = truncate_text
_safe_int = safe_int
_extract_message_token_usage = extract_message_token_usage
_extract_message_id = extract_message_id
_sum_usage_map = sum_usage_map


def _get_auth_redis():
    return auth_redis


require_auth_user = register_auth_routes(
    app,
    get_auth_redis=_get_auth_redis,
    auth_session_ttl_seconds=AUTH_SESSION_TTL_SECONDS,
    auth_user_key=_auth_user_key,
    auth_session_key=_auth_session_key,
    validate_username=_validate_username,
    validate_password=_validate_password,
    hash_password=_hash_password,
    verify_password=_verify_password,
)

register_sim_routes(
    app,
    sim_stream_dir=SIM_STREAM_DIR,
    sim_meta_file=SIM_META_FILE,
    sim_frame_file=SIM_FRAME_FILE,
    sim_frame_cache=_SIM_FRAME_CACHE,
)


# ========== 10. 接口定义 ==========

# 健康检查端点
@app.get("/api/health")
async def health_check():
    """后端健康检查"""
    health_info = {
        "status": "healthy",
        "agent_ready": active_agent is not None,
    }

    # 检查 Redis 连接
    redis_status = {"chat": False, "auth": False}
    try:
        if chat_redis:
            await chat_redis.ping()
            redis_status["chat"] = True
    except Exception:
        pass

    try:
        if auth_redis:
            await auth_redis.ping()
            redis_status["auth"] = True
    except Exception:
        pass

    health_info["redis"] = redis_status

    # 检查 MCP 服务（通过端口连接检测）
    mcp_status = {}
    try:
        import socket
        # PyBullet MCP (port 18001)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 18001))
            sock.close()
            mcp_status["pybullet"] = "online" if result == 0 else "offline"
        except Exception:
            mcp_status["pybullet"] = "offline"

        # Gazebo MCP (port 8002)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 8002))
            sock.close()
            mcp_status["gazebo"] = "online" if result == 0 else "offline"
        except Exception:
            mcp_status["gazebo"] = "offline"
    except Exception as e:
        mcp_status["error"] = str(e)

    health_info["mcp"] = mcp_status

    # 模型配置信息
    health_info["config"] = {
        "llm_model": config.get("llm", "unknown"),
        "llm_url": config.get("model_url", "unknown"),
    }

    return health_info


# 工具列表端点
@app.get("/api/tools")
async def list_tools():
    """返回所有可用的工具列表"""
    from tools import GeneralTool

    tools_info = []

    # 获取 GeneralTool 中的工具
    for func_name in GeneralTool.__all__:
        try:
            func = getattr(GeneralTool, func_name)
            # 获取工具描述
            tool_description = func.__doc__ or "无描述"
            # 提取第一行作为简介
            brief = tool_description.strip().split("\n")[0] if tool_description else "无描述"

            # 获取参数信息
            params = {}
            if hasattr(func, "args_schema") and func.args_schema:
                schema = func.args_schema.schema() if hasattr(func.args_schema, 'schema') else {}
                params = schema.get("properties", {})

            tools_info.append({
                "name": func_name,
                "brief": brief,
                "description": tool_description.strip(),
                "parameters": list(params.keys()),
            })
        except Exception as e:
            logger.warning(f"获取工具 {func_name} 信息失败: {e}")

    # 获取 MCP 服务工具（使用 fastmcp client）
    mcp_tools = []
    try:
        from fastmcp import Client

        # PyBullet MCP
        try:
            client = Client("http://localhost:18001/mcp")
            async with client:
                tools = await client.list_tools()
                for t in tools:
                    mcp_tools.append({
                        "name": t.name,
                        "source": "pybullet",
                        "description": t.description or "",
                    })
        except Exception as e:
            logger.warning(f"获取 PyBullet MCP 工具失败: {e}")

        # Gazebo MCP
        try:
            client = Client("http://localhost:8002/mcp")
            async with client:
                tools = await client.list_tools()
                for t in tools:
                    mcp_tools.append({
                        "name": t.name,
                        "source": "gazebo",
                        "description": t.description or "",
                    })
        except Exception as e:
            logger.warning(f"获取 Gazebo MCP 工具失败: {e}")
    except ImportError:
        logger.warning("fastmcp client 未安装")
    except Exception as e:
        logger.warning(f"MCP 工具获取失败: {e}")

    return {
        "local_tools": tools_info,
        "mcp_tools": mcp_tools,
        "total": len(tools_info) + len(mcp_tools),
    }


@app.get("/api/ping")
async def ping():
    return {
        "status": "ok",
        "agent_ready": active_agent is not None,  # 新增：返回agent状态
    }


@app.get("/api/messages")
async def get_messages(
    session_id: str = "default_session",
    limit: int = 100,
    current_user: dict = Depends(require_auth_user),
):
    if chat_redis is None:
        return []

    normalized_limit = max(1, min(limit, 500))
    key = _chat_history_key(current_user.get("uid", "unknown"), session_id)
    raw_items = await chat_redis.lrange(key, -normalized_limit, -1)

    messages = []
    for item in raw_items:
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                messages.append(parsed)
        except Exception:
            continue

    return messages


@app.get("/api/sessions")
async def get_sessions(
    limit: int = 50,
    current_user: dict = Depends(require_auth_user),
):
    if chat_redis is None:
        return []

    normalized_limit = max(1, min(limit, 500))
    ranked = await chat_redis.zrevrange(
        _chat_sessions_key(current_user.get("uid", "unknown")),
        0,
        normalized_limit - 1,
        withscores=True,
    )
    if not ranked:
        return []

    result = []
    user_id = current_user.get("uid", "unknown")
    pipe = chat_redis.pipeline()
    for session_id, _ in ranked:
        key = _chat_history_key(user_id, session_id)
        pipe.lindex(key, 0)
        pipe.lindex(key, -1)
    raw_items = await pipe.execute()

    idx = 0
    for session_id, score in ranked:
        first_raw = raw_items[idx]
        last_raw = raw_items[idx + 1]
        idx += 2
        title = ""
        preview = ""
        last_role = "assistant"
        if first_raw:
            try:
                first_msg = json.loads(first_raw)
                title = str(first_msg.get("text") or "").strip()
            except Exception:
                title = ""
        if last_raw:
            try:
                last_msg = json.loads(last_raw)
                preview = str(last_msg.get("text") or "")[:120]
                last_role = str(last_msg.get("role") or "assistant")
            except Exception:
                preview = ""

        if not title:
            title = preview or "新对话"

        result.append(
            {
                "session_id": session_id,
                "title": title,
                "updated_at": score,
                "preview": preview,
                "last_role": last_role,
            }
        )

    return result


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(require_auth_user),
):
    if chat_redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="聊天存储未就绪",
        )

    user_id = current_user.get("uid", "unknown")
    key = _chat_history_key(user_id, session_id)
    sessions_key = _chat_sessions_key(user_id)
    pipe = chat_redis.pipeline()
    pipe.delete(key)
    pipe.zrem(sessions_key, session_id)
    deleted_messages, removed_session = await pipe.execute()

    return {
        "ok": True,
        "deleted": bool(
            _safe_int(deleted_messages) > 0 or _safe_int(removed_session) > 0
        ),
        "session_id": session_id,
    }


@app.post("/api/chat/send")
async def chat_send(
    payload: ChatIn,
    current_user: dict = Depends(require_auth_user),
):
    user_message = payload.message or ""
    session_id = payload.session_id or "default_session"  # 使用会话ID或默认值
    enabled_tools = {
        str(t).strip()
        for t in (payload.enabled_tools or [])
        if str(t).strip()
    }
    # academic_search 已内置到 agent 工具中，始终启用
    academic_search_enabled = True
    web_search_enabled = "web_search" in enabled_tools  # 保留兼容旧版
    user_id = current_user.get("uid", "unknown")
    await _append_chat_message(user_id, session_id, "user", user_message)

    # 核心修正：使用全局的active_agent，而非初始空工具的agent
    if not active_agent:
        # 返回流式错误响应（保持格式统一）并添加防缓冲头
        return StreamingResponse(
            iter(
                [
                    json.dumps(
                        {"type": "error", "error": "Agent 未初始化完成，请检查日志"},
                        ensure_ascii=False,
                    )
                    + "\n"
                ]
            ),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-transform", "X-Accel-Buffering": "no"},
        )

    async def event_stream():
        prev_rag_disabled = os.environ.get("RAG_DISABLED")
        if prev_rag_disabled:
            os.environ["RAG_DISABLED"] = "1"
        main_latest_text = ""
        main_stream_text = ""
        thinking_stream_text = ""
        thinking_sent_text = ""
        in_think_tag = False
        think_tag_carry = ""
        thinking_truncated = False
        MAX_THINKING_CHARS = 1600
        last_planning_signature = ""
        usage_by_message: dict[str, dict[str, int]] = {}
        usage_by_agent_message: dict[str, dict[str, dict[str, int]]] = {}
        message_source_by_id: dict[str, str] = {}
        last_usage_signature = ""
        last_usage_by_agent_signature = ""
        last_status_signature = ""
        last_tool_output_signature = ""
        debug_msg_count = 0
        try:
            analysis_tool_names = set(getattr(AnalysisTool, "__all__", []))
            main_tool_names = set(getattr(GeneralTool, "__all__", []))

            def _resolve_agent_source(meta) -> str:
                if not isinstance(meta, dict):
                    return "main"
                parts = []
                for key in (
                    "langgraph_path",
                    "langgraph_node",
                    "langgraph_checkpoint_ns",
                    "checkpoint_ns",
                ):
                    value = meta.get(key)
                    if isinstance(value, (list, tuple)):
                        parts.extend(str(x).lower() for x in value)
                    elif value is not None:
                        parts.append(str(value).lower())
                text = " ".join(parts)
                if any(k in text for k in ("simulator", "pybullet", "gazebo")):
                    return "simulator"
                if any(k in text for k in ("data-analyzer", "analysis", "analyzer")):
                    return "analysis"
                return "main"

            def _resolve_tool_source(tool_name: str) -> str:
                name = str(tool_name or "").strip()
                if name == "write_todos":
                    return "main"
                if name in analysis_tool_names:
                    return "analysis"
                if name in main_tool_names:
                    return "main"
                return "simulator"

            def _compute_missing_delta(current_stream: str, latest_full: str) -> str:
                """Return only the unsent tail from latest_full relative to current_stream."""
                stream = _normalize_text(current_stream)
                latest = _normalize_text(latest_full)
                if not latest:
                    return ""
                if not stream:
                    return latest
                if latest == stream:
                    return ""
                if latest.startswith(stream):
                    return latest[len(stream) :]
                # If stream already contains latest (e.g. whitespace normalized differently),
                # do not emit any fallback to avoid duplicated full paragraphs.
                if latest in stream:
                    return ""
                # Longest suffix/prefix overlap fallback.
                max_k = min(len(stream), len(latest))
                overlap = 0
                for k in range(max_k, 0, -1):
                    if stream[-k:] == latest[:k]:
                        overlap = k
                        break
                return latest[overlap:]

            def _extract_tool_display_text(raw_text: str) -> str:
                text = _normalize_text(raw_text).strip()
                if not text:
                    return ""
                normalized_quotes = (
                    text.replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", "'")
                    .replace("’", "'")
                )
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(normalized_quotes)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        result = parsed.get("result") or parsed.get("message")
                        artifacts = parsed.get("artifacts")
                        if result is not None:
                            result_text = _normalize_text(result).strip()
                            if isinstance(artifacts, list) and artifacts:
                                cleaned = [
                                    _normalize_text(str(a)).strip()
                                    for a in artifacts
                                    if a is not None
                                ]
                                cleaned = [a for a in cleaned if a]
                                if cleaned:
                                    return f"{result_text} | artifacts: {', '.join(cleaned)}"
                            return result_text
                        # Common simulator return payload: format to readable summary.
                        if any(
                            k in parsed
                            for k in (
                                "final_position",
                                "velocity",
                                "final_velocity",
                                "status",
                            )
                        ):
                            parts = []
                            if parsed.get("status") is not None:
                                parts.append(f"status={parsed.get('status')}")
                            if parsed.get("final_position") is not None:
                                parts.append(f"final_position={parsed.get('final_position')}")
                            if parsed.get("velocity") is not None:
                                parts.append(f"velocity={parsed.get('velocity')}")
                            if parsed.get("final_velocity") is not None:
                                parts.append(f"final_velocity={parsed.get('final_velocity')}")
                            if parts:
                                return "仿真结果：" + ", ".join(parts)
                m = re.search(
                    r'"result"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"[a-zA-Z_]+"|}$)',
                    normalized_quotes,
                )
                if m:
                    return m.group(1).strip()
                return text

            def _extract_web_search_refs(raw_text: str):
                text = _normalize_text(raw_text).strip()
                if not text:
                    return []
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(text)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    results = parsed.get("results")
                    if not isinstance(results, list):
                        continue
                    refs = []
                    for item in results[:8]:
                        if not isinstance(item, dict):
                            continue
                        title = _normalize_text(item.get("title") or "Search Result")
                        url = _normalize_text(item.get("url") or "")
                        snippet = _normalize_text(item.get("snippet") or "")
                        if not url:
                            continue
                        refs.append(
                            {
                                "title": _truncate_text(title, max_len=120),
                                "url": url,
                                "snippet": _truncate_text(snippet, max_len=220),
                            }
                        )
                    return refs
                return []

            def _extract_academic_search_refs(raw_text: str):
                """提取 academic_search 结果"""
                text = _normalize_text(raw_text).strip()
                if not text:
                    return []
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(text)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    results = parsed.get("results")
                    if not isinstance(results, list):
                        continue
                    refs = []
                    for item in results[:8]:
                        if not isinstance(item, dict):
                            continue
                        # academic_search 返回的字段
                        title = _normalize_text(item.get("title") or "Academic Paper")
                        url = _normalize_text(item.get("url") or "")
                        abstract = _normalize_text(item.get("abstract") or "")
                        authors = _normalize_text(item.get("authors") or "")
                        year = item.get("year", "")
                        source = item.get("source", "")
                        if not url:
                            continue
                        refs.append(
                            {
                                "title": _truncate_text(title, max_len=120),
                                "url": url,
                                "snippet": _truncate_text(abstract, max_len=220),
                                "authors": authors,
                                "year": str(year),
                                "source": source,
                            }
                        )
                    return refs
                return []

            def _extract_search_refs(raw_text: str):
                """提取 unified search 结果（包含 web 和 paper 类型）"""
                text = _normalize_text(raw_text).strip()
                if not text:
                    return []
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(text)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    results = parsed.get("results")
                    if not isinstance(results, list):
                        continue
                    refs = []
                    for item in results[:8]:
                        if not isinstance(item, dict):
                            continue
                        title = _normalize_text(item.get("title") or "Search Result")
                        url = _normalize_text(item.get("url") or "")
                        if not url:
                            continue
                        # Paper type: has abstract/authors
                        if item.get("type") == "paper":
                            abstract = _normalize_text(item.get("abstract") or "")
                            authors = _normalize_text(item.get("authors") or "")
                            year = item.get("year", "")
                            source = item.get("source", "")
                            refs.append({
                                "title": _truncate_text(title, max_len=120),
                                "url": url,
                                "snippet": _truncate_text(abstract, max_len=220),
                                "authors": authors,
                                "year": str(year),
                                "source": source,
                            })
                        else:
                            # Web type: has snippet
                            snippet = _normalize_text(item.get("snippet") or "")
                            src = item.get("source", "")
                            refs.append({
                                "title": _truncate_text(title, max_len=120),
                                "url": url,
                                "snippet": _truncate_text(snippet, max_len=220),
                            })
                    return refs
                return []

            def _extract_rag_refs(raw_text: str):
                text = _normalize_text(raw_text).strip()
                if not text:
                    return []
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(text)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    refs = []
                    raw_refs = parsed.get("references")
                    if isinstance(raw_refs, list):
                        for item in raw_refs[:8]:
                            if not isinstance(item, dict):
                                continue
                            title = _normalize_text(item.get("label") or "Reference")
                            url = _normalize_text(item.get("url") or "")
                            if not url:
                                continue
                            refs.append(
                                {
                                    "title": _truncate_text(title, max_len=140),
                                    "url": url,
                                }
                            )
                    if refs:
                        return refs
                    raw_results = parsed.get("results")
                    if isinstance(raw_results, list):
                        for item in raw_results[:8]:
                            if not isinstance(item, dict):
                                continue
                            title = _normalize_text(
                                item.get("citation_label")
                                or item.get("doc_source")
                                or "Reference"
                            )
                            url = _normalize_text(item.get("citation_url") or "")
                            if not url:
                                continue
                            refs.append(
                                {
                                    "title": _truncate_text(title, max_len=140),
                                    "url": url,
                                }
                            )
                    return refs
                return []

            def _is_main_agent_message(meta) -> bool:
                if not isinstance(meta, dict):
                    return True
                node = str(meta.get("langgraph_node") or "")
                if node and node != "model":
                    return False
                path = meta.get("langgraph_path")
                # main graph model chunks are usually exactly this path;
                # subagent/internal chunks often have longer/deeper paths.
                if isinstance(path, (list, tuple)) and len(path) > 2:
                    return False
                return True

            runtime_tool_note = (
                "本轮可用工具：search（智能搜索，同时支持学术论文和网页）。请自主判断并调用 search。"
                if academic_search_enabled
                else "本轮禁用工具：search。"
            )
            input_messages = [
                {"role": "system", "content": runtime_tool_note},
                {"role": "user", "content": user_message},
            ]
            # 同时订阅 messages(增量token) 与 values(状态事件)，保证前端实时流式显示。
            print(f"[DEBUG] ====== Starting astream for user={user_id} session={session_id} ======")
            async for mode, event in active_agent.astream(
                {"messages": input_messages},
                stream_mode=["messages", "values"],
                config={"configurable": {"thread_id": f"{user_id}:{session_id}"}},
            ):
                if mode == "messages":
                    print(f"[DEBUG] ======== messages mode event ========")
                    try:
                        msg, _meta = event
                    except Exception:
                        msg = event
                        _meta = None
                    if DEBUG_STREAM_FIELDS and debug_msg_count < 6:
                        debug_msg_count += 1
                        if isinstance(msg, dict):
                            logger.info(
                                f"[stream-debug] msg(dict) keys={list(msg.keys())}"
                            )
                        else:
                            logger.info(
                                f"[stream-debug] msg(type={msg.__class__.__name__}) has content_blocks={hasattr(msg, 'content_blocks')} additional_kwargs={hasattr(msg, 'additional_kwargs')} response_metadata={hasattr(msg, 'response_metadata')}"
                            )
                    role_name = _normalize_message_role(msg)
                    source = _resolve_agent_source(_meta)
                    print(f"[DEBUG] role_name={role_name} source={source}")
                    if role_name in {"assistant", "ai"}:
                        msg_id = _extract_message_id(msg)
                        if msg_id:
                            message_source_by_id[msg_id] = source
                    if role_name in {"assistant", "ai"} and not _is_main_agent_message(_meta):
                        node = ""
                        path_text = ""
                        if isinstance(_meta, dict):
                            node = str(_meta.get("langgraph_node") or "")
                            raw_path = _meta.get("langgraph_path")
                            if isinstance(raw_path, (list, tuple)):
                                path_text = "/".join(str(x) for x in raw_path)
                            elif raw_path is not None:
                                path_text = str(raw_path)

                        target = ""
                        lowered_path = path_text.lower()
                        if "simulator" in lowered_path:
                            target = "simulator"
                        elif "data-analyzer" in lowered_path:
                            target = "data-analyzer"

                        if target:
                            status_text = f"已转交 {target} 执行，正在处理中..."
                        elif node and node != "model":
                            status_text = f"正在执行节点：{node}"
                        else:
                            status_text = "正在调用子代理执行任务..."

                        signature = f"subagent:{status_text}"
                        if signature != last_status_signature:
                            last_status_signature = signature
                            yield json.dumps(
                                {"type": "status", "text": status_text, "source": source},
                                ensure_ascii=False,
                            ) + "\n"

                    if role_name in {"assistant", "ai"}:
                        thinking_text = _extract_thinking_from_message(msg)
                        print(f"[DEBUG] role={role_name} thinking_text={repr(thinking_text[:100]) if thinking_text else ''}")
                        if thinking_text:
                            if len(thinking_text) > MAX_THINKING_CHARS:
                                thinking_text = thinking_text[:MAX_THINKING_CHARS]
                                thinking_truncated = True
                            if thinking_text.startswith(thinking_sent_text):
                                thinking_delta = thinking_text[
                                    len(thinking_sent_text) :
                                ]
                            else:
                                thinking_delta = thinking_text
                            if thinking_delta:
                                thinking_sent_text = thinking_text
                                thinking_stream_text = thinking_text
                                yield json.dumps(
                                    {
                                        "type": "thinking",
                                        "text": thinking_delta,
                                        "source": source,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"

                        delta = _extract_text_from_message(msg)
                        print(f"[DEBUG] delta={repr(delta[:200]) if delta else ''}")
                        if delta:
                            answer_delta, think_tag_delta, in_think_tag, think_tag_carry = (
                                _split_think_and_answer_delta(
                                    delta,
                                    in_think=in_think_tag,
                                    carry=think_tag_carry,
                                )
                            )

                            if think_tag_delta:
                                remain = MAX_THINKING_CHARS - len(thinking_stream_text)
                                if remain > 0:
                                    to_emit = think_tag_delta[:remain]
                                    thinking_stream_text += to_emit
                                    if to_emit:
                                        yield json.dumps(
                                            {
                                                "type": "thinking",
                                                "text": to_emit,
                                                "source": source,
                                            },
                                            ensure_ascii=False,
                                        ) + "\n"
                                if len(think_tag_delta) > max(remain, 0):
                                    thinking_truncated = True

                            if answer_delta:
                                print(f"[DEBUG] YIELDING DELTA: {repr(answer_delta[:100])} source={source}")
                                if source == "main":
                                    main_stream_text += answer_delta
                                yield json.dumps(
                                    {
                                        "type": "delta",
                                        "text": answer_delta,
                                        "source": source,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                    continue

                if mode != "values":
                    print(f"[DEBUG] non-messages/values mode: {mode}")
                    continue

                print(f"[DEBUG] ======== values mode event ========")
                messages = event.get("messages") if isinstance(event, dict) else None
                if isinstance(messages, list):
                    print(f"[DEBUG] values messages list len={len(messages)}")
                    for message in messages:
                        role_name = _normalize_message_role(message)
                        if role_name not in {"assistant", "ai"}:
                            continue
                        usage = _extract_message_token_usage(message)
                        if not usage:
                            continue
                        msg_id = _extract_message_id(message)
                        prev = usage_by_message.get(msg_id) or {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                        merged = {
                            "prompt_tokens": max(
                                _safe_int(prev.get("prompt_tokens")),
                                _safe_int(usage.get("prompt_tokens")),
                            ),
                            "completion_tokens": max(
                                _safe_int(prev.get("completion_tokens")),
                                _safe_int(usage.get("completion_tokens")),
                            ),
                            "total_tokens": max(
                                _safe_int(prev.get("total_tokens")),
                                _safe_int(usage.get("total_tokens")),
                            ),
                        }
                        if merged["total_tokens"] <= 0:
                            merged["total_tokens"] = (
                                merged["prompt_tokens"] + merged["completion_tokens"]
                            )
                        usage_by_message[msg_id] = merged

                    # Build per-agent usage attribution map from message-id/source cache.
                    for msg_id, merged_usage in usage_by_message.items():
                        source = message_source_by_id.get(msg_id, "main")
                        per_agent_map = usage_by_agent_message.setdefault(source, {})
                        prev_usage = per_agent_map.get(msg_id) or {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                        per_agent_map[msg_id] = {
                            "prompt_tokens": max(
                                _safe_int(prev_usage.get("prompt_tokens")),
                                _safe_int(merged_usage.get("prompt_tokens")),
                            ),
                            "completion_tokens": max(
                                _safe_int(prev_usage.get("completion_tokens")),
                                _safe_int(merged_usage.get("completion_tokens")),
                            ),
                            "total_tokens": max(
                                _safe_int(prev_usage.get("total_tokens")),
                                _safe_int(merged_usage.get("total_tokens")),
                            ),
                        }

                    usage_summary = _sum_usage_map(usage_by_message)
                    usage_signature = json.dumps(
                        usage_summary, ensure_ascii=False, sort_keys=True
                    )
                    if (
                        usage_signature != last_usage_signature
                        and usage_summary.get("total_tokens", 0) > 0
                    ):
                        last_usage_signature = usage_signature
                        yield json.dumps(
                            {
                                "type": "usage",
                                "usage": usage_summary,
                                "updated_at": time.time(),
                            },
                            ensure_ascii=False,
                        ) + "\n"

                    usage_by_agent_summary = {}
                    for agent_name, usage_map in usage_by_agent_message.items():
                        summary = _sum_usage_map(usage_map)
                        if summary.get("total_tokens", 0) > 0:
                            usage_by_agent_summary[agent_name] = summary
                    usage_by_agent_signature = json.dumps(
                        usage_by_agent_summary, ensure_ascii=False, sort_keys=True
                    )
                    if (
                        usage_by_agent_signature != last_usage_by_agent_signature
                        and usage_by_agent_summary
                    ):
                        last_usage_by_agent_signature = usage_by_agent_signature
                        logger.info(
                            "[token-usage][stream] user=%s session=%s usage_by_agent=%s",
                            user_id,
                            session_id,
                            usage_by_agent_summary,
                        )

                    # process all tool messages in current state so planning
                    # can update multiple times instead of only tracking the last one.
                    for message in messages:
                        role_msg = _normalize_message_role(message)
                        if role_msg != "tool":
                            continue
                        name_msg = _extract_message_name(message)
                        content_msg = None
                        if isinstance(message, dict):
                            content_msg = message.get("content") or message.get("text")
                        else:
                            content_msg = getattr(message, "content", None) or getattr(
                                message, "text", None
                            )

                        if name_msg == "write_todos":
                            planning_steps = _extract_planning_steps_from_write_todos(
                                content_msg
                            )
                            status_text = "正在规划执行步骤..."
                            signature = f"tool:{name_msg}:{status_text}"
                            if signature != last_status_signature:
                                last_status_signature = signature
                                yield json.dumps(
                                    {"type": "status", "text": status_text, "source": "main"},
                                    ensure_ascii=False,
                                ) + "\n"
                            if planning_steps:
                                signature = json.dumps(
                                    planning_steps, ensure_ascii=False, sort_keys=True
                                )
                                if signature != last_planning_signature:
                                    last_planning_signature = signature
                                    planning_payload = {
                                        "type": "planning",
                                        "plan": planning_steps,
                                        "updated_at": time.time(),
                                    }
                                    yield json.dumps(
                                        planning_payload, ensure_ascii=False
                                    ) + "\n"
                            # write_todos 属于 planning 源，不写入右侧工具时间轴，避免重复展示。
                            continue

                        # Timeline output disabled by product requirement.
                        tool_source = _resolve_tool_source(name_msg)
                        is_web_search_tool = name_msg == "web_search"
                        is_academic_search_tool = name_msg == "academic_search"
                        is_unified_search_tool = name_msg == "search"
                        is_rag_tool = name_msg == "qdrant_retrieve_context"
                        tool_status_text = (
                            "智能搜索中..."
                            if is_unified_search_tool
                            else "学术论文搜索中..."
                            if is_academic_search_tool
                            else "联网搜索中..."
                            if is_web_search_tool
                            else f"正在执行工具：{name_msg or 'tool'}"
                        )
                        signature = f"tool:{name_msg}:{_truncate_text(_normalize_text(content_msg), max_len=60)}"
                        if signature != last_status_signature:
                            last_status_signature = signature
                            yield json.dumps(
                                {
                                    "type": "status",
                                    "text": tool_status_text,
                                    "source": tool_source,
                                    "status_kind": "search" if (is_web_search_tool or is_academic_search_tool or is_unified_search_tool) else "tool",
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        tool_text = _normalize_text(content_msg)
                        if is_web_search_tool and tool_text:
                            refs = _extract_web_search_refs(tool_text)
                            if refs:
                                yield json.dumps(
                                    {
                                        "type": "web_search_results",
                                        "source": "main",
                                        "results": refs,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                        if is_academic_search_tool and tool_text:
                            refs = _extract_academic_search_refs(tool_text)
                            if refs:
                                yield json.dumps(
                                    {
                                        "type": "web_search_results",
                                        "source": "main",
                                        "results": refs,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                        if is_unified_search_tool and tool_text:
                            refs = _extract_search_refs(tool_text)
                            if refs:
                                yield json.dumps(
                                    {
                                        "type": "web_search_results",
                                        "source": "main",
                                        "results": refs,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                        if is_rag_tool and tool_text:
                            refs = _extract_rag_refs(tool_text)
                            if refs:
                                yield json.dumps(
                                    {
                                        "type": "rag_results",
                                        "source": "main",
                                        "results": refs,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                        if tool_text and tool_source in {"simulator", "analysis"}:
                            # Parse <think>...</think> in subagent/tool output to avoid leaking
                            # reasoning tags into final visible answer text.
                            think_matches = re.findall(
                                r"<think>([\s\S]*?)</think>", tool_text, flags=re.IGNORECASE
                            )
                            if think_matches:
                                tool_thinking_text = "\n".join(
                                    part.strip() for part in think_matches if part and part.strip()
                                ).strip()
                                if tool_thinking_text:
                                    tool_thinking_text = tool_thinking_text[:MAX_THINKING_CHARS]
                                    yield json.dumps(
                                        {
                                            "type": "thinking",
                                            "text": tool_thinking_text,
                                            "source": tool_source,
                                        },
                                        ensure_ascii=False,
                                    ) + "\n"
                            tool_text = re.sub(
                                r"<think>[\s\S]*?</think>",
                                "",
                                tool_text,
                                flags=re.IGNORECASE,
                            ).strip()
                            tool_text = _extract_tool_display_text(tool_text)

                            output_signature = (
                                f"{tool_source}:{name_msg}:"
                                f"{hashlib.md5(tool_text.encode('utf-8', errors='ignore')).hexdigest()}"
                            )
                            if output_signature != last_tool_output_signature:
                                last_tool_output_signature = output_signature
                                yield json.dumps(
                                    {
                                        "type": "delta",
                                        "text": _truncate_text(tool_text, max_len=600),
                                        "source": tool_source,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"

                try:
                    last = event["messages"][-1]
                except Exception:
                    last = None

                if last is None:
                    continue

                role = _normalize_message_role(last)
                name = _extract_message_name(last)
                content = None
                if isinstance(last, dict):
                    content = last.get("content") or last.get("text")
                else:
                    content = getattr(last, "content", None) or getattr(
                        last, "text", None
                    )

                # planning/timeline are handled above by scanning all tool messages.

                if role == "assistant" and content is not None:
                    main_latest_text = _normalize_text(content)

            # Fallback: if no/partial incremental output was emitted, flush final text.
            if main_latest_text and main_latest_text != main_stream_text:
                final_delta = _compute_missing_delta(main_stream_text, main_latest_text)
                if final_delta:
                    answer_delta, think_tag_delta, in_think_tag, think_tag_carry = (
                        _split_think_and_answer_delta(
                            final_delta, in_think=in_think_tag, carry=think_tag_carry
                        )
                    )
                    if think_tag_delta:
                        remain = MAX_THINKING_CHARS - len(thinking_stream_text)
                        if remain > 0:
                            to_emit = think_tag_delta[:remain]
                            thinking_stream_text += to_emit
                            if to_emit:
                                yield json.dumps(
                                    {
                                        "type": "thinking",
                                        "text": to_emit,
                                        "source": "main",
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                        if len(think_tag_delta) > max(remain, 0):
                            thinking_truncated = True
                    if answer_delta:
                        main_stream_text += answer_delta
                        yield json.dumps(
                            {"type": "delta", "text": answer_delta, "source": "main"},
                            ensure_ascii=False,
                        ) + "\n"

            if thinking_stream_text:
                yield json.dumps(
                    {"type": "thinking_done", "truncated": thinking_truncated},
                    ensure_ascii=False,
                ) + "\n"

            final_text = main_latest_text or main_stream_text
            print(f"[DEBUG] final_text={repr(final_text[:200]) if final_text else 'EMPTY'} main_latest_text={repr(main_latest_text[:100]) if main_latest_text else ''} main_stream_text={repr(main_stream_text[:100]) if main_stream_text else ''}")
            if final_text:
                await _append_chat_message(user_id, session_id, "assistant", final_text)
            final_usage_by_agent = {}
            for agent_name, usage_map in usage_by_agent_message.items():
                summary = _sum_usage_map(usage_map)
                if summary.get("total_tokens", 0) > 0:
                    final_usage_by_agent[agent_name] = summary
            if final_usage_by_agent:
                logger.info(
                    "[token-usage][final] user=%s session=%s usage_by_agent=%s",
                    user_id,
                    session_id,
                    final_usage_by_agent,
                )
            # 发送完成信号
            print(f"[DEBUG] ====== STREAM COMPLETE, yielding done ======")
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(
                f"调用Agent出错：{str(e)}", exc_info=True
            )  # 修正：使用uvicorn logger
            await _append_chat_message(
                user_id, session_id, "assistant", f"[后端错误] 处理请求失败：{str(e)}"
            )
            if thinking_stream_text:
                yield json.dumps(
                    {"type": "thinking_done", "truncated": thinking_truncated},
                    ensure_ascii=False,
                ) + "\n"
            yield json.dumps(
                {"type": "error", "error": f"处理请求失败：{str(e)}"},
                ensure_ascii=False,
            ) + "\n"
        finally:
            if prev_rag_disabled:
                if prev_rag_disabled is None:
                    os.environ.pop("RAG_DISABLED", None)
                else:
                    os.environ["RAG_DISABLED"] = prev_rag_disabled

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-transform", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
