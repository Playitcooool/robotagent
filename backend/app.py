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
from prompts.context_loader import ContextLoader
from prompts.MainAgentPrompt import build_system_prompt_with_context
import logging
import json
import yaml
from tools import GeneralTool
from tools import AnalysisTool
import time
from langchain_openai import ChatOpenAI
import re
import hashlib
import httpx
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
from backend.utils.retry import with_retry_async
from backend.workflow_utils import (
    extract_chat_history_text,
    normalize_intent_result,
    restore_env_var,
    text_claims_simulator_execution,
    text_waits_for_simulator_result,
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
DEBUG_STREAM_FIELDS = os.environ.get("DEBUG_STREAM_FIELDS", "0") == "1"
DEBUG_STREAM_TOKENS = os.environ.get("DEBUG_STREAM_TOKENS", "0") == "1"

# ========== 2. 全局变量定义（关键：提前声明active_agent） ==========
active_agent = None  # 默认 agent：不加载联网搜索工具
active_search_agent = None  # 联网搜索 agent：仅用户打开开关时使用
agent_checkpointer_cm = None
chat_redis: Optional[Redis] = None
auth_redis: Optional[Redis] = None
context_loader: Optional[ContextLoader] = None  # Context 文件加载器
CHAT_REDIS_URL = os.environ["CHAT_REDIS_URL"]
AUTH_REDIS_URL = os.environ["AUTH_REDIS_URL"]
CHAT_HISTORY_PREFIX = "robotagent:chat:messages"
CHAT_HISTORY_MAX_LEN = int(os.environ.get("CHAT_HISTORY_MAX_LEN", "200"))
CHAT_SESSIONS_ZSET_PREFIX = "robotagent:chat:sessions"
PENDING_ACTION_PREFIX = "robotagent:chat:pending_action"
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default


def _truncate_for_prefill(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[...context truncated for faster prefill...]"


# ========== Intent detection via small LLM ==========
_INTENT_PROMPT = """判断用户意图。根据用户消息、最近对话和待确认动作，输出一个JSON：
{"simulator_required": true/false, "execution_confirmed": true/false, "pending_action_response": "confirmed|modify|rejected|unclear", "confidence": 0.0-1.0}

规则：
- simulator_required: 当前用户消息是否需要机器人仿真/执行/物理环境工具。
- execution_confirmed: 只有待确认动作为 simulator，且当前用户明确同意执行该待确认动作时才为 true。
- pending_action_response: 当存在待确认动作时，判断当前用户是在确认执行、修改参数、拒绝执行，还是含义不清。
- 如果待确认动作是“无”，execution_confirmed 必须为 false；用户说“执行一个任务”只表示需要先规划/补齐参数/请求确认。
- 如果信息不足、用户只是在补充参数、或没有待确认动作，不要把 execution_confirmed 设为 true。
- 不要依赖固定关键词；结合语义和上下文判断。

只输出JSON，不要其他内容。/no_think"""


async def detect_intent(
    user_message: str,
    recent_context: str = "",
    pending_action: str = "",
) -> dict:
    """Use small LLM to classify user intent for simulator routing."""
    url = config.get("intent_model_url", config["model_url"])
    model = config.get("intent_llm", "Qwen:Qwen3-0.6B")
    api_key = config.get("intent_api_key", config.get("api_key", "no_need"))

    messages = [
        {"role": "system", "content": _INTENT_PROMPT},
        {
            "role": "user",
            "content": (
                f"待确认动作：{pending_action or '无'}\n\n"
                f"最近对话：{recent_context[:1200]}\n\n"
                f"当前用户消息：{user_message}"
            ),
        },
    ]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "temperature": 0, "max_tokens": 50},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Parse JSON from response (handle possible markdown wrapping)
            content = content.strip("`").removeprefix("json").strip()
            result = json.loads(content)
            return normalize_intent_result(
                result,
                pending_action,
                user_message=user_message,
                recent_context=recent_context,
            )
    except Exception as e:
        logger.warning(f"Intent detection failed, using conservative fallback: {e}")
        return normalize_intent_result(
            {
                "simulator_required": pending_action == "simulator",
                "execution_confirmed": False,
                "pending_action_response": "unclear",
                "confidence": 0.0,
            },
            pending_action,
            user_message=user_message,
            recent_context=recent_context,
        )


# ========== 4. 加载工具函数（保留并添加日志） ==========
def get_tools(enabled_general_tools: set[str] | None = None):
    try:
        general_tools = []
        subagent_tools = []
        if enabled_general_tools is None:
            configured_tools = os.environ.get("MAIN_GENERAL_TOOLS", "current_time")
            enabled_general_tools = {
                name.strip()
                for name in configured_tools.split(",")
                if name.strip()
            }
        for func_name in GeneralTool.__all__:
            if enabled_general_tools and func_name not in enabled_general_tools:
                continue
            function = getattr(GeneralTool, func_name)
            general_tools.append(function)

        # 合并工具
        all_tools = general_tools + subagent_tools
        logger.info(
            "总工具数量：%s；MainAgent 通用工具：%s",
            len(all_tools),
            [getattr(t, "name", str(t)) for t in general_tools],
        )
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
    global active_agent, active_search_agent, agent_checkpointer_cm, chat_redis, auth_redis, context_loader  # 关联全局变量
    try:
        # Redis connection with retry (指数退避 + jitter)
        async def connect_chat_redis():
            return Redis.from_url(CHAT_REDIS_URL, decode_responses=True)
        chat_redis = await with_retry_async(
            connect_chat_redis,
            max_retries=5,
            base_delay=0.5,
            retryable_exceptions=(ConnectionError, OSError),
        )
        await chat_redis.ping()
        logger.info(f"Chat history Redis 已连接: {CHAT_REDIS_URL}")

        async def connect_auth_redis():
            return Redis.from_url(AUTH_REDIS_URL, decode_responses=True)
        auth_redis = await with_retry_async(
            connect_auth_redis,
            max_retries=5,
            base_delay=0.5,
            retryable_exceptions=(ConnectionError, OSError),
        )
        await auth_redis.ping()
        logger.info(f"Auth Redis 已连接: {AUTH_REDIS_URL}")

        # 初始化 ContextLoader（加载 robot_context.md）
        context_loader = ContextLoader()
        logger.info("ContextLoader 已初始化")

        # 加载默认工具。联网 search 工具只挂到 search agent，避免每轮都进入 prefill。
        all_tools = get_tools({"current_time"})
        search_tools = get_tools({"current_time", "search"})
        if not all_tools:
            logger.warning("未加载到任何工具，agent将使用空工具列表")

        # 加载 experience（从 agent_experiences.json）。
        # 为了降低 prefill，默认不注入；需要时设置 ENABLE_EXPERIENCE=1。
        # WITHOUT_EXPERIENCE=1 继续作为强制关闭开关。
        inject_experiences = (
            _env_bool("ENABLE_EXPERIENCE", False)
            and not _env_bool("WITHOUT_EXPERIENCE", False)
        )
        if inject_experiences:
            experiences = _load_agent_experiences()
            logger.info(f"已加载 {len(experiences)} 条 agent experiences")
        else:
            experiences = []
            logger.info("已跳过 experience 注入；设置 ENABLE_EXPERIENCE=1 可开启")

        subagents = list(
            await init_subagents(
                experiences=experiences,
                max_experiences_in_subagent=_env_int(
                    "MAX_EXPERIENCES_IN_SUBAGENT", 3, minimum=0
                ),
            )
        )

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
        # 注意：MainAgent 只接收 main_agent 的经验，不接收 simulation/analysis 的经验，避免干扰
        if experiences:
            exp_suffix = build_experience_suffix(experiences, agent_filter="main")
        else:
            exp_suffix = ""

        # 加载 robot_context.md
        context = context_loader.load_context() if context_loader else ""
        context = _truncate_for_prefill(
            context,
            _env_int("ROBOT_CONTEXT_MAX_CHARS", 1200, minimum=0),
        )
        logger.info(f"已加载 robot_context.md: {len(context)} 字符" if context else "未找到 robot_context.md")

        runtime_system_prompt = build_system_prompt_with_context(
            MainAgentPrompt.SYSTEM_PROMPT,
            context,
            _truncate_for_prefill(
                exp_suffix,
                _env_int("MAIN_EXPERIENCE_MAX_CHARS", 1500, minimum=0),
            )
        ) + prompt_suffix

        # 创建带工具的agent（核心修正）
        DB_URI = "redis://localhost:6379"
        checkpointer = None
        if _env_bool("AGENT_CHECKPOINT_ENABLED", True):
            agent_checkpointer_cm = AsyncRedisSaver.from_conn_string(DB_URI)
            checkpointer = await agent_checkpointer_cm.__aenter__()
            await checkpointer.asetup()
        active_agent = create_deep_agent(
            model=chatBot,
            tools=all_tools,
            system_prompt=runtime_system_prompt,
            subagents=subagents,
            checkpointer=checkpointer,
        )
        active_search_agent = create_deep_agent(
            model=chatBot,
            tools=search_tools,
            system_prompt=runtime_system_prompt,
            subagents=subagents,
            checkpointer=checkpointer,
        )
        logger.info("Agent 初始化成功！")
    except Exception as e:
        logger.error(f"Agent 初始化失败：{str(e)}", exc_info=True)
        active_agent = None
        active_search_agent = None
        if agent_checkpointer_cm is not None:
            await agent_checkpointer_cm.__aexit__(None, None, None)
            agent_checkpointer_cm = None
        if chat_redis is not None:
            await chat_redis.aclose()
            chat_redis = None
        if auth_redis is not None:
            await auth_redis.aclose()
            auth_redis = None


@app.on_event("shutdown")
async def shutdown_event():
    global chat_redis, auth_redis, agent_checkpointer_cm
    if agent_checkpointer_cm is not None:
        await agent_checkpointer_cm.__aexit__(None, None, None)
        agent_checkpointer_cm = None
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


def _pending_action_key(user_id: str, session_id: str) -> str:
    return f"{PENDING_ACTION_PREFIX}:{user_id}:{session_id}"


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


async def _get_pending_action(user_id: str, session_id: str) -> str:
    if chat_redis is None:
        return ""
    raw = await chat_redis.get(_pending_action_key(user_id, session_id))
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        payload = json.loads(raw)
    except Exception:
        return ""
    return _normalize_text(payload.get("action") or "").strip()


async def _set_pending_action(
    user_id: str,
    session_id: str,
    action: str,
    summary: str = "",
) -> None:
    if chat_redis is None:
        return
    payload = {
        "action": action,
        "summary": _truncate_text(_normalize_text(summary).strip(), max_len=600),
        "created_at": time.time(),
    }
    await chat_redis.set(
        _pending_action_key(user_id, session_id),
        json.dumps(payload, ensure_ascii=False),
        ex=3600,
    )


async def _clear_pending_action(user_id: str, session_id: str) -> None:
    if chat_redis is not None:
        await chat_redis.delete(_pending_action_key(user_id, session_id))

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
    search_enabled = bool(enabled_tools & {"search", "academic_search", "web_search"})
    user_id = current_user.get("uid", "unknown")

    # Use small LLM for intent detection; pending actions carry plan/execute state.
    recent_context_messages = []
    if chat_redis:
        key = _chat_history_key(user_id, session_id)
        recent = await chat_redis.lrange(key, -12, -1)
        for item in reversed(recent or []):
            try:
                parsed = json.loads(item)
                content = _normalize_text(extract_chat_history_text(parsed)).strip()
                if not content:
                    continue
                role = parsed.get("role")
                if role in {"assistant", "user"}:
                    recent_context_messages.append(f"{role}: {content}")
            except Exception:
                continue
    recent_context = "\n\n".join(recent_context_messages[:6])
    pending_action = await _get_pending_action(user_id, session_id)

    intent = await detect_intent(user_message, recent_context, pending_action)
    simulator_required = intent["simulator_required"]
    simulator_execution_confirmed = intent["execution_confirmed"]
    if simulator_execution_confirmed and pending_action != "simulator":
        logger.warning(
            "Intent execution confirmation ignored without pending simulator action user=%s session=%s pending_action=%s pending_response=%s confidence=%s message_preview=%r",
            user_id,
            session_id,
            pending_action or "none",
            intent.get("pending_action_response"),
            intent.get("confidence"),
            _truncate_text(user_message, max_len=220),
        )
        simulator_execution_confirmed = False
    if (
        simulator_execution_confirmed
        and pending_action == "simulator"
        and not simulator_required
    ):
        logger.warning(
            "Intent inconsistency normalized: execution_confirmed=true but simulator_required=false user=%s session=%s pending_action=%s pending_response=%s confidence=%s message_preview=%r",
            user_id,
            session_id,
            pending_action or "none",
            intent.get("pending_action_response"),
            intent.get("confidence"),
            _truncate_text(user_message, max_len=220),
        )
        simulator_required = True
    logger.info(
        "Intent detected: simulator_required=%s execution_confirmed=%s pending_action=%s pending_response=%s confidence=%s context_chars=%s",
        simulator_required,
        simulator_execution_confirmed,
        pending_action or "none",
        intent.get("pending_action_response"),
        intent.get("confidence"),
        len(recent_context),
    )

    await _append_chat_message(user_id, session_id, "user", user_message)
    if simulator_execution_confirmed:
        await _clear_pending_action(user_id, session_id)
    elif pending_action == "simulator" and intent.get("pending_action_response") == "rejected":
        await _clear_pending_action(user_id, session_id)

    # 核心修正：使用全局的active_agent，而非初始空工具的agent
    selected_agent = active_search_agent if search_enabled else active_agent
    if not selected_agent:
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
        current_planning_steps = []
        current_status_text = ""
        current_status_source = "main"
        simulator_activity_seen = False
        simulator_tool_activity_seen = False
        simulator_task_call_seen = False
        last_simulator_output_text = ""
        stream_started_at = time.time()
        pending_answer_whitespace_by_source: dict[str, str] = {}
        debug_msg_count = 0
        stream_message_events = 0
        stream_values_events = 0
        tool_names_seen: set[str] = set()
        simulator_tool_names_seen: set[str] = set()
        debug_stream_token_counts: dict[str, int] = {}
        try:
            analysis_tool_names = set(getattr(AnalysisTool, "__all__", []))
            main_tool_names = set(getattr(GeneralTool, "__all__", []))

            def _debug_stream_token(
                *,
                source: str,
                channel: str,
                text: str,
                msg_id: str = "",
                role: str = "",
                name: str = "",
            ) -> None:
                if not DEBUG_STREAM_TOKENS or not text:
                    return
                normalized_source = source or "main"
                key = f"{normalized_source}:{channel}"
                debug_stream_token_counts[key] = debug_stream_token_counts.get(key, 0) + 1
                logger.info(
                    "[stream-token] user=%s session=%s source=%s channel=%s seq=%s role=%s name=%s msg_id=%s chars=%s text=%r",
                    user_id,
                    session_id,
                    normalized_source,
                    channel,
                    debug_stream_token_counts[key],
                    role or "-",
                    name or "-",
                    msg_id or "-",
                    len(text),
                    text,
                )

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

            def _sim_frame_received_since_start() -> bool:
                if not SIM_META_FILE.exists() or not SIM_FRAME_FILE.exists():
                    return False
                try:
                    frame_mtime = SIM_FRAME_FILE.stat().st_mtime
                except Exception:
                    frame_mtime = 0.0
                try:
                    with open(SIM_META_FILE, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta_ts = float(meta.get("timestamp") or 0.0)
                except Exception:
                    meta_ts = 0.0
                return max(frame_mtime, meta_ts) >= stream_started_at - 1.0

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

            def _build_planning_payload(
                plan=None,
                status_text: str = "",
                active_source: str = "main",
                is_active: bool = False,
            ) -> dict:
                return {
                    "type": "planning",
                    "plan": plan if isinstance(plan, list) else [],
                    "updated_at": time.time(),
                    "status_text": _normalize_text(status_text).strip(),
                    "active_source": active_source or "main",
                    "is_active": bool(is_active),
                }

            def _message_has_simulator_task_call(message) -> bool:
                candidates = []
                if isinstance(message, dict):
                    candidates.extend(
                        [
                            message.get("tool_calls"),
                            message.get("invalid_tool_calls"),
                            (message.get("additional_kwargs") or {}).get("tool_calls")
                            if isinstance(message.get("additional_kwargs"), dict)
                            else None,
                        ]
                    )
                else:
                    candidates.extend(
                        [
                            getattr(message, "tool_calls", None),
                            getattr(message, "invalid_tool_calls", None),
                        ]
                    )
                    additional_kwargs = getattr(message, "additional_kwargs", None)
                    if isinstance(additional_kwargs, dict):
                        candidates.append(additional_kwargs.get("tool_calls"))

                for raw_calls in candidates:
                    if not isinstance(raw_calls, list):
                        continue
                    for call in raw_calls:
                        if not isinstance(call, dict):
                            continue
                        name = call.get("name")
                        args = call.get("args") or call.get("arguments")
                        function = call.get("function")
                        if isinstance(function, dict):
                            name = name or function.get("name")
                            args = args or function.get("arguments")
                        if name != "task":
                            continue
                        parsed_args = args
                        if isinstance(args, str):
                            parsed_args = None
                            for parser in (json.loads, ast.literal_eval):
                                try:
                                    parsed_args = parser(args)
                                    break
                                except Exception:
                                    continue
                        if (
                            isinstance(parsed_args, dict)
                            and parsed_args.get("subagent_type") == "simulator"
                        ):
                            return True
                return False

            runtime_tool_note = (
                "本轮可用工具：search（智能搜索，同时支持学术论文和网页）。请自主判断并调用 search。"
                if search_enabled
                else "本轮禁用工具：search。"
            )
            if simulator_required and simulator_execution_confirmed:
                runtime_tool_note += (
                    "\n本轮是机器人仿真/执行任务：必须立即调用 task(subagent_type=\"simulator\", "
                    "description=\"...\")。禁止只用文字声称已委托、正在执行或等待结果。"
                )
            elif simulator_required:
                runtime_tool_note += (
                    "\n本轮是机器人仿真规划请求：先给出简短计划、你选择的关键参数和确认问题；"
                    "未收到用户明确确认前不要调用 simulator。"
                    "注意：你必须在最终回复中输出文字给用户（计划摘要+确认问题），不能只调用工具就结束。"
                )
            input_messages = [
                {"role": "system", "content": runtime_tool_note},
                {"role": "user", "content": user_message},
            ]
            # 同时订阅 messages(增量token) 与 values(状态事件)，保证前端实时流式显示。
            logger.info(
                "[chat-stream] start user=%s session=%s search=%s simulator_required=%s execution_confirmed=%s",
                user_id,
                session_id,
                search_enabled,
                simulator_required,
                simulator_execution_confirmed,
            )
            async for mode, event in selected_agent.astream(
                {"messages": input_messages},
                stream_mode=["messages", "values"],
                config={"configurable": {"thread_id": f"{user_id}:{session_id}"}},
            ):
                if mode == "messages":
                    stream_message_events += 1
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
                    if source == "simulator":
                        simulator_activity_seen = True
                    if role_name in {"assistant", "ai"}:
                        msg_id = _extract_message_id(msg)
                        if msg_id:
                            message_source_by_id[msg_id] = source
                        if _message_has_simulator_task_call(msg):
                            simulator_task_call_seen = True
                            logger.info(
                                "[chat-stream] simulator task call detected user=%s session=%s message_id=%s",
                                user_id,
                                session_id,
                                msg_id or "unknown",
                            )
                            status_text = "正在调用 simulator 执行仿真..."
                            signature = f"task-call:{status_text}"
                            if signature != last_status_signature:
                                last_status_signature = signature
                                current_status_text = status_text
                                current_status_source = "simulator"
                                yield json.dumps(
                                    {
                                        "type": "status",
                                        "text": status_text,
                                        "source": "simulator",
                                        "status_kind": "tool",
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                                yield json.dumps(
                                    _build_planning_payload(
                                        current_planning_steps,
                                        current_status_text,
                                        current_status_source,
                                        True,
                                    ),
                                    ensure_ascii=False,
                                ) + "\n"
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
                            current_status_text = status_text
                            current_status_source = source
                            yield json.dumps(
                                {"type": "status", "text": status_text, "source": source},
                                ensure_ascii=False,
                            ) + "\n"
                            yield json.dumps(
                                _build_planning_payload(
                                    current_planning_steps,
                                    current_status_text,
                                    current_status_source,
                                    True,
                                ),
                                ensure_ascii=False,
                            ) + "\n"

                    if role_name in {"assistant", "ai"}:
                        thinking_text = _extract_thinking_from_message(msg)
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
                                _debug_stream_token(
                                    source=source,
                                    channel="thinking",
                                    text=thinking_delta,
                                    msg_id=msg_id or "",
                                    role=role_name,
                                )
                                yield json.dumps(
                                    {
                                        "type": "thinking",
                                        "text": thinking_delta,
                                        "source": source,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"

                        delta = _extract_text_from_message(msg)
                        if delta:
                            _debug_stream_token(
                                source=source,
                                channel="raw",
                                text=delta,
                                msg_id=msg_id or "",
                                role=role_name,
                            )
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
                                        _debug_stream_token(
                                            source=source,
                                            channel="think-tag",
                                            text=to_emit,
                                            msg_id=msg_id or "",
                                            role=role_name,
                                        )
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

                            suppress_confirmed_main_text = (
                                simulator_required
                                and simulator_execution_confirmed
                                and source == "main"
                            )
                            if answer_delta and not suppress_confirmed_main_text:
                                if not answer_delta.strip():
                                    pending_answer_whitespace_by_source[source] = (
                                        pending_answer_whitespace_by_source.get(source, "")
                                        + answer_delta
                                    )
                                    continue
                                emit_delta = (
                                    pending_answer_whitespace_by_source.pop(source, "")
                                    + answer_delta
                                )
                                if source == "main":
                                    main_stream_text += emit_delta
                                _debug_stream_token(
                                    source=source,
                                    channel="answer",
                                    text=emit_delta,
                                    msg_id=msg_id or "",
                                    role=role_name,
                                )
                                yield json.dumps(
                                    {
                                        "type": "delta",
                                        "text": emit_delta,
                                        "source": source,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"

                    # Emit real-time status for tool messages from subagents
                    if role_name == "tool" and source in {"simulator", "analysis"}:
                        tool_name = _extract_message_name(msg)
                        if tool_name and tool_name != "task":
                            simulator_activity_seen = True
                            if source == "simulator":
                                simulator_tool_activity_seen = True
                                simulator_tool_names_seen.add(tool_name)
                            tool_names_seen.add(tool_name)
                            logger.info(
                                "[chat-stream] subagent tool message user=%s session=%s source=%s tool=%s",
                                user_id,
                                session_id,
                                source,
                                tool_name,
                            )
                            tool_status_text = f"正在执行：{tool_name}"
                            signature = f"subtool:{tool_name}"
                            if signature != last_status_signature:
                                last_status_signature = signature
                                current_status_text = tool_status_text
                                current_status_source = source
                                yield json.dumps(
                                    {"type": "status", "text": tool_status_text, "source": source},
                                    ensure_ascii=False,
                                ) + "\n"
                                yield json.dumps(
                                    _build_planning_payload(
                                        current_planning_steps,
                                        current_status_text,
                                        current_status_source,
                                        True,
                                    ),
                                    ensure_ascii=False,
                                ) + "\n"

                    continue

                if mode != "values":
                    logger.info(
                        "[chat-stream] ignored stream mode user=%s session=%s mode=%s",
                        user_id,
                        session_id,
                        mode,
                    )
                    continue

                stream_values_events += 1
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
                                current_status_text = status_text
                                current_status_source = "main"
                                yield json.dumps(
                                    {"type": "status", "text": status_text, "source": "main"},
                                    ensure_ascii=False,
                                ) + "\n"
                                yield json.dumps(
                                    _build_planning_payload(
                                        current_planning_steps,
                                        current_status_text,
                                        current_status_source,
                                        True,
                                    ),
                                    ensure_ascii=False,
                                ) + "\n"
                            if planning_steps:
                                signature = json.dumps(
                                    planning_steps, ensure_ascii=False, sort_keys=True
                                )
                                if signature != last_planning_signature:
                                    last_planning_signature = signature
                                    current_planning_steps = planning_steps
                                    planning_payload = _build_planning_payload(
                                        current_planning_steps,
                                        current_status_text,
                                        current_status_source,
                                        True,
                                    )
                                    yield json.dumps(
                                        planning_payload, ensure_ascii=False
                                    ) + "\n"
                            # write_todos 属于 planning 源，不写入右侧工具时间轴，避免重复展示。
                            continue

                        # Timeline output disabled by product requirement.
                        tool_source = _resolve_tool_source(name_msg)
                        if tool_source == "simulator":
                            simulator_activity_seen = True
                            if name_msg != "task":
                                simulator_tool_activity_seen = True
                                simulator_tool_names_seen.add(name_msg or "unknown")
                        if name_msg:
                            tool_names_seen.add(name_msg)
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
                            else "仿真子代理执行中..."
                            if name_msg == "task"
                            else f"正在执行：{name_msg or 'tool'}"
                        )
                        signature = f"tool:{name_msg}:{_truncate_text(_normalize_text(content_msg), max_len=60)}"
                        if signature != last_status_signature:
                            last_status_signature = signature
                            current_status_text = tool_status_text
                            current_status_source = tool_source
                            yield json.dumps(
                                {
                                    "type": "status",
                                    "text": tool_status_text,
                                    "source": tool_source,
                                    "status_kind": "search" if (is_web_search_tool or is_academic_search_tool or is_unified_search_tool) else "tool",
                                },
                                ensure_ascii=False,
                            ) + "\n"
                            yield json.dumps(
                                _build_planning_payload(
                                    current_planning_steps,
                                    current_status_text,
                                    current_status_source,
                                    True,
                                ),
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
                                    _debug_stream_token(
                                        source=tool_source,
                                        channel="tool-thinking",
                                        text=tool_thinking_text,
                                        role=role_msg,
                                        name=name_msg or "",
                                    )
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
                            if tool_source == "simulator" and tool_text:
                                last_simulator_output_text = tool_text
                                logger.info(
                                    "[chat-stream] simulator tool output user=%s session=%s tool=%s output_chars=%s output_preview=%r",
                                    user_id,
                                    session_id,
                                    name_msg or "unknown",
                                    len(tool_text),
                                    _truncate_text(tool_text, max_len=220),
                                )

                            output_signature = (
                                f"{tool_source}:{name_msg}:"
                                f"{hashlib.md5(tool_text.encode('utf-8', errors='ignore')).hexdigest()}"
                            )
                            if output_signature != last_tool_output_signature:
                                last_tool_output_signature = output_signature
                                display_tool_text = _truncate_text(tool_text, max_len=600)
                                _debug_stream_token(
                                    source=tool_source,
                                    channel="tool-output",
                                    text=display_tool_text,
                                    role=role_msg,
                                    name=name_msg or "",
                                )
                                yield json.dumps(
                                    {
                                        "type": "delta",
                                        "text": display_tool_text,
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
            skip_main_fallback = simulator_required and simulator_execution_confirmed
            if main_latest_text and main_latest_text != main_stream_text and not skip_main_fallback:
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
                                _debug_stream_token(
                                    source="main",
                                    channel="fallback-think-tag",
                                    text=to_emit,
                                    role="assistant",
                                )
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
                        if not answer_delta.strip():
                            pending_answer_whitespace_by_source["main"] = (
                                pending_answer_whitespace_by_source.get("main", "")
                                + answer_delta
                            )
                            answer_delta = ""
                    if answer_delta:
                        emit_delta = (
                            pending_answer_whitespace_by_source.pop("main", "")
                            + answer_delta
                        )
                        main_stream_text += emit_delta
                        _debug_stream_token(
                            source="main",
                            channel="fallback-answer",
                            text=emit_delta,
                            role="assistant",
                        )
                        yield json.dumps(
                            {"type": "delta", "text": emit_delta, "source": "main"},
                            ensure_ascii=False,
                        ) + "\n"

            if thinking_stream_text:
                yield json.dumps(
                    {"type": "thinking_done", "truncated": thinking_truncated},
                    ensure_ascii=False,
                ) + "\n"

            final_text = main_latest_text or main_stream_text
            final_text_stripped = final_text.strip()
            sim_frame_received = _sim_frame_received_since_start()
            waiting_for_simulator_result = bool(
                simulator_required
                and simulator_execution_confirmed
                and (
                    simulator_tool_activity_seen
                    or simulator_task_call_seen
                    or sim_frame_received
                )
                and final_text_stripped
                and text_waits_for_simulator_result(final_text_stripped)
            )
            if waiting_for_simulator_result:
                if last_simulator_output_text:
                    final_text = (
                        "结论：仿真执行已完成。\n"
                        f"关键结果：{_truncate_text(last_simulator_output_text, max_len=700)}\n"
                        "下一步：可查看右侧最新帧或继续要求分析结果。"
                    )
                elif sim_frame_received:
                    final_text = (
                        "结论：仿真执行已完成，已收到仿真画面帧。\n"
                        "关键结果：右侧显示最新仿真帧；simulator 未返回额外文本摘要。\n"
                        "下一步：可继续要求分析轨迹或导出结果。"
                    )
                else:
                    final_text = (
                        "simulator task 已调用，但尚未收到画面帧或 MCP 工具文本结果。"
                        "请检查 simulator 子代理/MCP 服务日志。"
                    )
                final_text_stripped = final_text.strip()
                if final_text_stripped and final_text != main_stream_text:
                    main_stream_text = final_text
                    _debug_stream_token(
                        source="main",
                        channel="final-rewrite",
                        text=final_text,
                        role="assistant",
                    )
                    yield json.dumps(
                        {"type": "delta", "text": final_text, "source": "main"},
                        ensure_ascii=False,
                    ) + "\n"
            if (
                simulator_required
                and simulator_execution_confirmed
                and last_simulator_output_text
                and not simulator_tool_activity_seen
            ):
                diagnostic_text = (
                    "simulator 子代理已响应，但未收到仿真画面帧或 MCP 物理仿真工具输出。\n"
                    f"子代理结果：{_truncate_text(last_simulator_output_text, max_len=700)}"
                )
                if diagnostic_text.strip() != final_text_stripped:
                    final_text = diagnostic_text
                    final_text_stripped = final_text.strip()
                    main_stream_text = final_text
                    _debug_stream_token(
                        source="main",
                        channel="diagnostic",
                        text=diagnostic_text,
                        role="assistant",
                    )
                    yield json.dumps(
                        {"type": "delta", "text": diagnostic_text, "source": "main"},
                        ensure_ascii=False,
                    ) + "\n"
            elif (
                simulator_required
                and simulator_execution_confirmed
                and simulator_tool_activity_seen
                and not sim_frame_received
            ):
                diagnostic_text = "未收到画面帧；已保留 simulator 文本结果和工具状态用于诊断。"
                if diagnostic_text not in final_text_stripped:
                    final_text = (final_text_stripped + "\n" + diagnostic_text).strip()
                    final_text_stripped = final_text.strip()
                    main_stream_text = final_text
                    _debug_stream_token(
                        source="main",
                        channel="diagnostic",
                        text="\n" + diagnostic_text,
                        role="assistant",
                    )
                    yield json.dumps(
                        {"type": "delta", "text": "\n" + diagnostic_text, "source": "main"},
                        ensure_ascii=False,
                    ) + "\n"
            logger.info(
                "[chat-stream] final user=%s session=%s message_events=%s values_events=%s task_call_seen=%s simulator_activity_seen=%s simulator_tool_activity_seen=%s tools=%s simulator_tools=%s final_chars=%s final_preview=%r",
                user_id,
                session_id,
                stream_message_events,
                stream_values_events,
                simulator_task_call_seen,
                simulator_activity_seen,
                simulator_tool_activity_seen,
                sorted(tool_names_seen),
                sorted(simulator_tool_names_seen),
                len(final_text_stripped),
                _truncate_text(final_text_stripped, max_len=300),
            )
            misleading_simulator_claim = text_claims_simulator_execution(
                final_text_stripped
            )
            if (
                simulator_required
                and simulator_execution_confirmed
                and not simulator_tool_activity_seen
            ):
                if simulator_task_call_seen or last_simulator_output_text:
                    logger.warning(
                        "[chat-stream] simulator task responded without MCP frame/tool user=%s session=%s task_call_seen=%s output_chars=%s frame_received=%s",
                        user_id,
                        session_id,
                        simulator_task_call_seen,
                        len(last_simulator_output_text),
                        sim_frame_received,
                    )
                else:
                    no_tool_error = (
                        "已收到执行确认，但没有触发 simulator 工具调用。"
                        "请重试，或检查 simulator 子代理/MCP 服务是否可用。"
                    )
                    logger.warning(
                        "[chat-stream] confirmed execution without simulator MCP tool user=%s session=%s task_call_seen=%s tools=%s final_preview=%r",
                        user_id,
                        session_id,
                        simulator_task_call_seen,
                        sorted(tool_names_seen),
                        _truncate_text(final_text_stripped, max_len=300),
                    )
                    await _append_chat_message(user_id, session_id, "assistant", no_tool_error)
                    yield json.dumps(
                        {"type": "error", "error": no_tool_error},
                        ensure_ascii=False,
                    ) + "\n"
                    return
            if (
                simulator_execution_confirmed
                and not simulator_tool_activity_seen
                and not simulator_task_call_seen
                and not last_simulator_output_text
                and misleading_simulator_claim
            ):
                no_tool_error = (
                    "模型声称已委托或正在执行 simulator，但没有实际触发工具调用。"
                    "请先确认计划，确认后我会调用 simulator 执行。"
                )
                logger.warning(
                    "[chat-stream] misleading simulator claim user=%s session=%s task_call_seen=%s tools=%s final_preview=%r",
                    user_id,
                    session_id,
                    simulator_task_call_seen,
                    sorted(tool_names_seen),
                    _truncate_text(final_text_stripped, max_len=300),
                )
                await _append_chat_message(user_id, session_id, "assistant", no_tool_error)
                yield json.dumps(
                    {"type": "error", "error": no_tool_error},
                    ensure_ascii=False,
                ) + "\n"
                return

            if final_text_stripped:
                await _append_chat_message(user_id, session_id, "assistant", final_text)
                if simulator_required and not simulator_execution_confirmed:
                    await _set_pending_action(
                        user_id,
                        session_id,
                        "simulator",
                        final_text_stripped,
                    )
            elif current_planning_steps and simulator_required and not simulator_execution_confirmed:
                # Model only called write_todos but produced no text reply; synthesize one.
                steps_text = "\n".join(
                    f"  {i+1}. {s.get('content', '')}" for i, s in enumerate(current_planning_steps)
                )
                fallback_text = f"已为您规划以下执行步骤：\n{steps_text}\n\n确认执行吗？"
                final_text = fallback_text
                main_stream_text = fallback_text
                yield json.dumps(
                    {"type": "delta", "text": fallback_text, "source": "main"},
                    ensure_ascii=False,
                ) + "\n"
                await _append_chat_message(user_id, session_id, "assistant", fallback_text)
                await _set_pending_action(user_id, session_id, "simulator", fallback_text)
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
            if current_planning_steps or current_status_text:
                final_status_text = current_status_text
                final_status_source = current_status_source
                final_steps = current_planning_steps
                if (
                    simulator_required
                    and simulator_execution_confirmed
                    and (simulator_tool_activity_seen or sim_frame_received)
                ):
                    final_status_text = (
                        "仿真执行完成"
                        if sim_frame_received
                        else "仿真执行完成，未收到画面帧"
                    )
                    final_status_source = "simulator"
                    final_steps = [
                        {**step, "status": "completed"}
                        for step in current_planning_steps
                    ]
                elif (
                    simulator_required
                    and simulator_execution_confirmed
                    and (simulator_task_call_seen or last_simulator_output_text)
                ):
                    final_status_text = "simulator 子代理已响应，未收到画面帧/MCP 工具输出"
                    final_status_source = "simulator"
                yield json.dumps(
                    _build_planning_payload(
                        final_steps,
                        final_status_text,
                        final_status_source,
                        False,
                    ),
                    ensure_ascii=False,
                ) + "\n"
            # 发送完成信号
            logger.info(
                "[chat-stream] done user=%s session=%s final_persisted=%s planning_status=%r",
                user_id,
                session_id,
                bool(final_text_stripped),
                current_status_text,
            )
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
            restore_env_var("RAG_DISABLED", prev_rag_disabled)

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
