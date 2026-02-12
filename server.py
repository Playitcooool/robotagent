from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
import base64
from deepagents import create_deep_agent
from tools.SubAgentTool import init_subagents
import logging
import json
import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from tools import GeneralTool
import asyncio
from langchain_openai import ChatOpenAI
import os
from prompts import MainAgentPrompt
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import os

os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
# Shared realtime frame location written by mcp/mcp_server.py
# Default path points to repo-mounted directory so host and docker can share files.
DEFAULT_SIM_STREAM_DIR = (Path(__file__).resolve().parent / "mcp" / ".sim_stream").resolve()
SIM_STREAM_DIR = Path(
    os.environ.get("PYBULLET_STREAM_DIR", str(DEFAULT_SIM_STREAM_DIR))
).resolve()
SIM_META_FILE = SIM_STREAM_DIR / "latest.json"
SIM_FRAME_FILE = SIM_STREAM_DIR / "latest.png"
# ========== 1. 日志配置（确保输出到Uvicorn控制台） ==========
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# ========== 2. 全局变量定义（关键：提前声明active_agent） ==========
active_agent = None  # 全局agent，启动事件中初始化
with open("config/config.yml", "r", encoding="utf-8") as f:
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
    global active_agent  # 关联全局变量
    try:
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


# ========== 8. CORS中间件 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 9. 请求模型 ==========
class ChatIn(BaseModel):
    message: str
    session_id: str = None  # 新增：会话ID，用于维持对话状态


# ========== 10. 接口定义 ==========
@app.get("/api/ping")
async def ping():
    return {
        "status": "ok",
        "agent_ready": active_agent is not None,  # 新增：返回agent状态
    }


@app.get("/api/messages")
async def get_messages():
    sample = [
        {"id": 1, "role": "assistant", "text": "示例对话：你好，我是 RobotAgent。"},
        {"id": 2, "role": "user", "text": "请帮我查一下最新的论文。"},
    ]
    return sample


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
        with open(SIM_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {"status": "error", "has_frame": False, "error": f"meta read failed: {e}"}

    try:
        with open(SIM_FRAME_FILE, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        return {
            "status": "error",
            "has_frame": False,
            "error": f"frame read failed: {e}",
        }

    return {
        "status": "done" if meta.get("done") else "running",
        "has_frame": True,
        "run_id": meta.get("run_id"),
        "task": meta.get("task"),
        "step": meta.get("step"),
        "total_steps": meta.get("total_steps"),
        "done": bool(meta.get("done")),
        "timestamp": meta.get("timestamp"),
        "image_url": f"data:image/png;base64,{image_b64}",
    }


@app.get("/api/sim/latest-frame")
async def get_latest_sim_frame():
    return _load_latest_frame_payload()


@app.get("/api/sim/stream")
async def stream_sim_frames(request: Request, since: float = 0.0):
    """SSE endpoint that actively pushes latest simulation frames."""

    async def event_stream():
        last_ts = float(since or 0.0)
        idle_ticks = 0

        while True:
            if await request.is_disconnected():
                break

            payload = _load_latest_frame_payload()
            if payload.get("has_frame"):
                current_ts = float(payload.get("timestamp") or 0.0)
                if current_ts > last_ts:
                    last_ts = current_ts
                    idle_ticks = 0
                    yield f"event: frame\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    idle_ticks += 1
            else:
                idle_ticks += 1

            # keep-alive every ~5s (100 * 50ms) so proxies won't close idle SSE
            if idle_ticks >= 100:
                idle_ticks = 0
                yield "event: ping\ndata: {}\n\n"

            await asyncio.sleep(0.05)

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
async def chat_send(payload: ChatIn):
    user_message = payload.message or ""
    session_id = payload.session_id or "default_session"  # 使用会话ID或默认值

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
        try:
            # 调用全局active_agent的流式接口，传递配置包含thread_id
            async for event in active_agent.astream(
                {"messages": [{"role": "user", "content": user_message}]},
                stream_mode="values",
                config={"configurable": {"thread_id": session_id}},
            ):
                try:
                    last = event["messages"][-1]
                    print(last)
                except Exception:
                    last = None

                if last is None:
                    continue

                content = None
                if isinstance(last, dict):
                    content = last.get("content") or last.get("text")
                else:
                    content = getattr(last, "content", None) or getattr(
                        last, "text", None
                    )

                if content is not None:
                    payload = {"type": "delta", "text": content}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"

            # 发送完成信号
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(
                f"调用Agent出错：{str(e)}", exc_info=True
            )  # 修正：使用uvicorn logger
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
        "server:app",  # 如果文件名是 server.py 就写 server:app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
