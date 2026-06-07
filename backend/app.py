import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from backend.auth_utils import (
    hash_password,
    validate_password,
    validate_username,
    verify_password,
)
from backend.routes_auth import register_auth_routes
from backend.routes_sim import register_sim_routes
from backend.utils.retry import with_retry_async

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,host.docker.internal")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,host.docker.internal")
os.environ.setdefault("CHAT_REDIS_URL", "redis://127.0.0.1:6379/1")
os.environ.setdefault("AUTH_REDIS_URL", "redis://127.0.0.1:6379/2")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SIM_STREAM_DIR = (ROOT_DIR / "mcp" / ".sim_stream").resolve()
SIM_STREAM_DIR = Path(
    os.environ.get("PYBULLET_STREAM_DIR", str(DEFAULT_SIM_STREAM_DIR))
).resolve()
SIM_META_FILE = SIM_STREAM_DIR / "latest.json"
SIM_FRAME_FILE = SIM_STREAM_DIR / "latest.png"
SIM_REPLAY_DIR = SIM_STREAM_DIR / "replay"
_SIM_FRAME_CACHE = {
    "meta_mtime": 0.0,
    "frame_mtime": 0.0,
    "payload": {"status": "idle", "has_frame": False},
}

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

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

chat_redis: Optional[Redis] = None
auth_redis: Optional[Redis] = None

app = FastAPI(title="RobotAgent Legacy Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_config() -> dict:
    config_path = ROOT_DIR / "config" / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader) or {}


def _chat_history_key(user_id: str, session_id: str) -> str:
    safe = (session_id or "default_session").strip() or "default_session"
    return f"{CHAT_HISTORY_PREFIX}:{user_id}:{safe}"


def _chat_sessions_key(user_id: str) -> str:
    return f"{CHAT_SESSIONS_ZSET_PREFIX}:{user_id}"


def _auth_user_key(username: str) -> str:
    return f"{AUTH_USER_PREFIX}:{username.lower()}"


def _auth_session_key(token: str) -> str:
    return f"{AUTH_SESSION_PREFIX}:{token}"


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_auth_redis():
    return auth_redis


@app.on_event("startup")
async def startup_event():
    global chat_redis, auth_redis

    async def connect_chat_redis():
        return Redis.from_url(CHAT_REDIS_URL, decode_responses=True)

    chat_redis = await with_retry_async(
        connect_chat_redis,
        max_retries=5,
        base_delay=0.5,
        retryable_exceptions=(ConnectionError, OSError),
    )
    await chat_redis.ping()
    logger.info("Chat history Redis connected: %s", CHAT_REDIS_URL)

    async def connect_auth_redis():
        return Redis.from_url(AUTH_REDIS_URL, decode_responses=True)

    auth_redis = await with_retry_async(
        connect_auth_redis,
        max_retries=5,
        base_delay=0.5,
        retryable_exceptions=(ConnectionError, OSError),
    )
    await auth_redis.ping()
    logger.info("Auth Redis connected: %s", AUTH_REDIS_URL)


@app.on_event("shutdown")
async def shutdown_event():
    global chat_redis, auth_redis
    if chat_redis is not None:
        await chat_redis.aclose()
        chat_redis = None
    if auth_redis is not None:
        await auth_redis.aclose()
        auth_redis = None


require_auth_user = register_auth_routes(
    app,
    get_auth_redis=_get_auth_redis,
    auth_session_ttl_seconds=AUTH_SESSION_TTL_SECONDS,
    auth_user_key=_auth_user_key,
    auth_session_key=_auth_session_key,
    validate_username=validate_username,
    validate_password=validate_password,
    hash_password=hash_password,
    verify_password=verify_password,
)

register_sim_routes(
    app,
    sim_stream_dir=SIM_STREAM_DIR,
    sim_meta_file=SIM_META_FILE,
    sim_frame_file=SIM_FRAME_FILE,
    sim_replay_dir=SIM_REPLAY_DIR,
    sim_frame_cache=_SIM_FRAME_CACHE,
)


@app.get("/api/health")
async def health_check():
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

    cfg = _load_config()
    return {
        "status": "healthy",
        "agent_ready": False,
        "legacy_backend": True,
        "redis": redis_status,
        "config": {
            "llm_model": cfg.get("llm", "unknown"),
            "llm_url": cfg.get("model_url", "unknown"),
        },
    }


@app.get("/api/ping")
async def ping():
    return {"status": "ok", "agent_ready": False, "legacy_backend": True}


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
    user_id = current_user.get("uid", "unknown")
    ranked = await chat_redis.zrevrange(
        _chat_sessions_key(user_id),
        0,
        normalized_limit - 1,
        withscores=True,
    )
    if not ranked:
        return []

    pipe = chat_redis.pipeline()
    for session_id, _ in ranked:
        key = _chat_history_key(user_id, session_id)
        pipe.lindex(key, 0)
        pipe.lindex(key, -1)
    raw_items = await pipe.execute()

    result = []
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
    pipe = chat_redis.pipeline()
    pipe.delete(_chat_history_key(user_id, session_id))
    pipe.zrem(_chat_sessions_key(user_id), session_id)
    deleted_messages, removed_session = await pipe.execute()
    return {
        "ok": True,
        "deleted": bool(_safe_int(deleted_messages) > 0 or _safe_int(removed_session) > 0),
        "session_id": session_id,
    }


@app.post("/api/chat/send")
async def removed_chat_send():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Python LangChain agent has been removed. Use the Pi agent gateway.",
    )


@app.post("/v1/chat/completions")
async def removed_openai_chat_completions():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Python LangChain OpenAI-compatible agent has been removed. Use the Pi agent gateway.",
    )


@app.get("/v1/models")
async def removed_models():
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Model listing is served by the Pi agent gateway.",
    )
