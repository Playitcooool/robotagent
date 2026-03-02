from fastapi import FastAPI
from fastapi import Request
from fastapi import Depends, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path
import base64
import ast
from deepagents import create_deep_agent
from tools.SubAgentTool import init_subagents
import logging
import json
import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from tools import GeneralTool
import asyncio
import time
from langchain_openai import ChatOpenAI
import os
import re
import hmac
import hashlib
import secrets
from prompts import MainAgentPrompt
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redis.asyncio import Redis
from typing import Optional
from backend.schemas import ChatIn, AuthRegisterIn, AuthLoginIn
from backend.auth_utils import (
    validate_username,
    validate_password,
    hash_password,
    verify_password,
)
from backend.stream_utils import (
    normalize_text,
    extract_text_from_message,
    normalize_message_role,
    extract_message_name,
    extract_planning_steps_from_write_todos,
    truncate_text,
    safe_int,
    extract_message_token_usage,
    extract_message_id,
    sum_usage_map,
)

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
AUTH_SESSION_TTL_SECONDS = int(os.environ.get("AUTH_SESSION_TTL_SECONDS", str(30 * 24 * 3600)))
PASSWORD_PBKDF2_ITERATIONS = int(os.environ.get("AUTH_PASSWORD_PBKDF2_ITERATIONS", "200000"))
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
    model=config["llm"]["chat"],
    api_key="no_need",
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
        subagents = list(await init_subagents())
        # 创建带工具的agent（核心修正）
        DB_URI = "redis://localhost:6379"
        async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.asetup()
            active_agent = create_deep_agent(
                model=chatBot,
                tools=all_tools,
                system_prompt=MainAgentPrompt.SYSTEM_PROMPT,
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


def _validate_username(username: str) -> str:
    cleaned = (username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名需为3-32位，仅支持字母/数字/._-",
        )
    return cleaned


def _validate_password(password: str) -> str:
    cleaned = password or ""
    if len(cleaned) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码长度至少6位",
        )
    return cleaned


def _hash_password(password: str, salt_b64: Optional[str] = None):
    if salt_b64:
        salt = base64.b64decode(salt_b64.encode("ascii"))
    else:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_PBKDF2_ITERATIONS,
    )
    return (
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, salt_b64: str, expected_hash_b64: str) -> bool:
    _, actual_hash = _hash_password(password, salt_b64=salt_b64)
    return hmac.compare_digest(actual_hash, expected_hash_b64)


async def _get_auth_user(authorization: Optional[str]) -> dict:
    if auth_redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认证服务未就绪",
        )

    header = (authorization or "").strip()
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
        )

    token = header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效",
        )

    session_raw = await auth_redis.get(_auth_session_key(token))
    if not session_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
        )
    try:
        session = json.loads(session_raw)
        if not isinstance(session, dict):
            raise ValueError("invalid session")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录会话无效",
        )

    # Sliding session window for easier debugging.
    await auth_redis.expire(_auth_session_key(token), AUTH_SESSION_TTL_SECONDS)
    return {"token": token, "uid": session.get("uid"), "username": session.get("username")}


async def require_auth_user(authorization: Optional[str] = Header(default=None)) -> dict:
    return await _get_auth_user(authorization)


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
_normalize_message_role = normalize_message_role
_extract_message_name = extract_message_name
_extract_planning_steps_from_write_todos = extract_planning_steps_from_write_todos
_truncate_text = truncate_text
_safe_int = safe_int
_extract_message_token_usage = extract_message_token_usage
_extract_message_id = extract_message_id
_sum_usage_map = sum_usage_map


# ========== 10. 接口定义 ==========
@app.get("/api/ping")
async def ping():
    return {
        "status": "ok",
        "agent_ready": active_agent is not None,  # 新增：返回agent状态
    }


@app.post("/api/auth/register")
async def auth_register(payload: AuthRegisterIn):
    if auth_redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认证服务未就绪",
        )

    username = _validate_username(payload.username)
    password = _validate_password(payload.password)
    user_key = _auth_user_key(username)
    existing = await auth_redis.get(user_key)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在",
        )

    salt_b64, password_hash_b64 = _hash_password(password)
    created_at = time.time()
    user_doc = {
        "uid": secrets.token_hex(12),
        "username": username,
        "password_salt": salt_b64,
        "password_hash": password_hash_b64,
        "created_at": created_at,
    }
    await auth_redis.set(user_key, json.dumps(user_doc, ensure_ascii=False))

    token = secrets.token_urlsafe(32)
    session_doc = {
        "uid": user_doc["uid"],
        "username": user_doc["username"],
        "created_at": created_at,
    }
    await auth_redis.setex(
        _auth_session_key(token), AUTH_SESSION_TTL_SECONDS, json.dumps(session_doc, ensure_ascii=False)
    )
    return {
        "token": token,
        "user": {"uid": user_doc["uid"], "username": user_doc["username"]},
        "expires_in": AUTH_SESSION_TTL_SECONDS,
    }


@app.post("/api/auth/login")
async def auth_login(payload: AuthLoginIn):
    if auth_redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认证服务未就绪",
        )

    username = _validate_username(payload.username)
    password = _validate_password(payload.password)
    user_raw = await auth_redis.get(_auth_user_key(username))
    if not user_raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    try:
        user_doc = json.loads(user_raw)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户数据损坏",
        )

    ok = _verify_password(password, user_doc.get("password_salt", ""), user_doc.get("password_hash", ""))
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    token = secrets.token_urlsafe(32)
    session_doc = {
        "uid": user_doc["uid"],
        "username": user_doc["username"],
        "created_at": time.time(),
    }
    await auth_redis.setex(
        _auth_session_key(token), AUTH_SESSION_TTL_SECONDS, json.dumps(session_doc, ensure_ascii=False)
    )
    return {
        "token": token,
        "user": {"uid": user_doc["uid"], "username": user_doc["username"]},
        "expires_in": AUTH_SESSION_TTL_SECONDS,
    }


@app.get("/api/auth/me")
async def auth_me(current_user: dict = Depends(require_auth_user)):
    return {"user": {"uid": current_user.get("uid"), "username": current_user.get("username")}}


@app.post("/api/auth/logout")
async def auth_logout(current_user: dict = Depends(require_auth_user)):
    if auth_redis is not None:
        await auth_redis.delete(_auth_session_key(current_user.get("token", "")))
    return {"ok": True}


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
        _chat_sessions_key(current_user.get("uid", "unknown")), 0, normalized_limit - 1, withscores=True
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
        "deleted": bool(_safe_int(deleted_messages) > 0 or _safe_int(removed_session) > 0),
        "session_id": session_id,
    }


@app.get("/api/sim/debug")
async def sim_debug():
    return {
        "stream_dir": str(SIM_STREAM_DIR),
        "meta_exists": SIM_META_FILE.exists(),
        "frame_exists": SIM_FRAME_FILE.exists(),
    }


def _load_latest_frame_payload():
    if not SIM_META_FILE.exists() or not SIM_FRAME_FILE.exists():
        return {"status": "idle", "has_frame": False}

    try:
        meta_mtime = SIM_META_FILE.stat().st_mtime
        frame_mtime = SIM_FRAME_FILE.stat().st_mtime
    except Exception:
        meta_mtime = 0.0
        frame_mtime = 0.0

    cached = _SIM_FRAME_CACHE
    if meta_mtime == cached.get("meta_mtime") and frame_mtime == cached.get("frame_mtime"):
        return cached.get("payload", {"status": "idle", "has_frame": False})

    try:
        with open(SIM_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {"status": "error", "has_frame": False, "error": f"meta read failed: {e}"}

    payload = {
        "status": "done" if meta.get("done") else "running",
        "has_frame": True,
        "run_id": meta.get("run_id"),
        "task": meta.get("task"),
        "step": meta.get("step"),
        "total_steps": meta.get("total_steps"),
        "done": bool(meta.get("done")),
        "timestamp": meta.get("timestamp"),
        "image_url": f"/api/sim/latest.png?ts={meta.get('timestamp')}",
    }
    _SIM_FRAME_CACHE["meta_mtime"] = meta_mtime
    _SIM_FRAME_CACHE["frame_mtime"] = frame_mtime
    _SIM_FRAME_CACHE["payload"] = payload
    return payload


@app.get("/api/sim/latest-frame")
async def get_latest_sim_frame():
    return _load_latest_frame_payload()


@app.get("/api/sim/latest.png")
async def get_latest_sim_png():
    if not SIM_FRAME_FILE.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="frame not found")
    return FileResponse(SIM_FRAME_FILE, media_type="image/png")


@app.get("/api/sim/stream")
async def stream_sim_frames(request: Request, since: float = 0.0):
    """SSE endpoint that actively pushes latest simulation frames."""

    async def event_stream():
        last_ts = float(since or 0.0)
        idle_ticks = 0
        last_emit_ts = 0.0

        while True:
            if await request.is_disconnected():
                break

            payload = _load_latest_frame_payload()
            if payload.get("has_frame"):
                current_ts = float(payload.get("timestamp") or 0.0)
                if current_ts > last_ts:
                    last_ts = current_ts
                    idle_ticks = 0
                    last_emit_ts = time.time()
                    yield f"event: frame\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    idle_ticks += 1
            else:
                idle_ticks += 1

            # keep-alive every ~5s (100 * 50ms) so proxies won't close idle SSE
            if idle_ticks >= 100:
                idle_ticks = 0
                yield "event: ping\ndata: {}\n\n"

            now = time.time()
            is_running = payload.get("status") == "running"
            if is_running and now - last_emit_ts < 2.0:
                sleep_s = 0.05
            elif idle_ticks > 200:
                sleep_s = 0.5
            else:
                sleep_s = 0.2
            await asyncio.sleep(sleep_s)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/send")
async def chat_send(
    payload: ChatIn,
    current_user: dict = Depends(require_auth_user),
):
    user_message = payload.message or ""
    session_id = payload.session_id or "default_session"  # 使用会话ID或默认值
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
        assistant_latest_text = ""
        assistant_stream_text = ""
        last_planning_signature = ""
        last_timeline_signature = ""
        usage_by_message: dict[str, dict[str, int]] = {}
        last_usage_signature = ""
        try:
            # 同时订阅 messages(增量token) 与 values(状态事件)，保证前端实时流式显示。
            async for mode, event in active_agent.astream(
                {"messages": [{"role": "user", "content": user_message}]},
                stream_mode=["messages", "values"],
                config={"configurable": {"thread_id": f"{user_id}:{session_id}"}},
            ):
                if mode == "messages":
                    try:
                        msg, _meta = event
                    except Exception:
                        msg = event
                    role_name = _normalize_message_role(msg)
                    if role_name in {"assistant", "ai"}:
                        delta = _extract_text_from_message(msg)
                        if delta:
                            assistant_stream_text += delta
                            yield json.dumps(
                                {"type": "delta", "text": delta}, ensure_ascii=False
                            ) + "\n"
                    continue

                if mode != "values":
                    continue

                messages = event.get("messages") if isinstance(event, dict) else None
                if isinstance(messages, list):
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

                    usage_summary = _sum_usage_map(usage_by_message)
                    usage_signature = json.dumps(
                        usage_summary, ensure_ascii=False, sort_keys=True
                    )
                    if usage_signature != last_usage_signature and usage_summary.get(
                        "total_tokens", 0
                    ) > 0:
                        last_usage_signature = usage_signature
                        yield json.dumps(
                            {
                                "type": "usage",
                                "usage": usage_summary,
                                "updated_at": time.time(),
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

                if role == "tool" and name == "write_todos":
                    planning_steps = _extract_planning_steps_from_write_todos(content)
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
                            yield json.dumps(planning_payload, ensure_ascii=False) + "\n"

                if role == "tool":
                    tool_text = _normalize_text(content)
                    timeline_item = {
                        "kind": "tool",
                        "title": name or "tool",
                        "detail": _truncate_text(tool_text, max_len=220),
                    }
                    timeline_signature = json.dumps(
                        timeline_item, ensure_ascii=False, sort_keys=True
                    )
                    if timeline_signature != last_timeline_signature:
                        last_timeline_signature = timeline_signature
                        yield json.dumps(
                            {
                                "type": "timeline",
                                "item": timeline_item,
                                "updated_at": time.time(),
                            },
                            ensure_ascii=False,
                        ) + "\n"

                if role == "assistant" and content is not None:
                    assistant_latest_text = _normalize_text(content)

            final_text = assistant_latest_text or assistant_stream_text
            if final_text:
                await _append_chat_message(
                    user_id, session_id, "assistant", final_text
                )
            # 发送完成信号
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(
                f"调用Agent出错：{str(e)}", exc_info=True
            )  # 修正：使用uvicorn logger
            await _append_chat_message(
                user_id, session_id, "assistant", f"[后端错误] 处理请求失败：{str(e)}"
            )
            yield json.dumps(
                {"type": "error", "error": f"处理请求失败：{str(e)}"},
                ensure_ascii=False,
            ) + "\n"

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
