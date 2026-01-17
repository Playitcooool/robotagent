from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import json

# Import the existing agent and chat bot from main.py (if available)
try:
    # main.py defines `agent` (an agent wrapper) and `chatBot` (the raw ChatOllama client)
    from main import agent, chatBot
except Exception:
    agent = None
    chatBot = None

# If no top-level agent is available, try to create a lightweight Ollama-based agent as a fallback.
ollama_agent = None
if agent is None:
    try:
        from langchain_ollama import ChatOllama
        from langchain.agents import create_agent
        import os

        if chatBot is None:
            chatBot = ChatOllama(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            )
        ollama_agent = create_agent(model=chatBot, tools=[])
    except Exception:
        ollama_agent = None

app = FastAPI()
# Allow frontend dev server to call this API; restrict in production as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str


@app.get("/api/ping")
async def ping():
    return {"status": "ok"}


@app.get("/api/messages")
async def get_messages():
    """Return a small list of recent messages for the frontend sidebar.

    This is a lightweight compatibility endpoint used by the dev frontend.
    """
    sample = [
        {"id": 1, "role": "assistant", "text": "示例对话：你好，我是 RobotAgent。"},
        {"id": 2, "role": "user", "text": "请帮我查一下最新的论文。"},
    ]
    return sample


@app.post("/api/chat/send")
async def chat_send(payload: ChatIn):
    """Accepts JSON { message: str } and streams newline-delimited JSON objects.

    Each line is a JSON object such as:
      {"type":"delta","text":"partial or full assistant text"}
      {"type":"done"}
      {"type":"error","error":"..."}

    The frontend should read the response body as a stream and parse each newline-delimited JSON object incrementally.
    """
    user_message = payload.message or ""

    # Prefer an existing orchestrating agent, otherwise fall back to the ollama-only agent
    active_agent = agent or ollama_agent
    if not active_agent:
        # Return a simple JSON reply instead of a stream for compatibility
        return {
            "reply": "[model-unavailable] 无法访问 Ollama 模型或代理未配置。请检查 Ollama 服务和环境变量 OLLAMA_BASE_URL/OLLAMA_MODEL。"
        }

    def event_stream():
        try:
            # Iterate the agent stream and yield newline-delimited JSON objects
            for event in active_agent.stream(
                {"messages": [{"role": "user", "content": user_message}]},
                stream_mode="values",
            ):
                try:
                    last = event["messages"][-1]
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
                    # send the assistant text as an incremental update
                    payload = {"type": "delta", "text": content}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"

            # signal completion
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            logging.exception("Error while invoking agent")
            yield json.dumps(
                {"type": "error", "error": str(e)}, ensure_ascii=False
            ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# uvicorn server:app --reload --host 0.0.0.0 --port 8000 to start backend
