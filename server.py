from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import the existing agent from main.py
try:
    from main import agent
except Exception:
    agent = None

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


@app.post("/api/chat/send")
async def chat_send(payload: ChatIn):
    """Accepts JSON { message: str } and returns { reply: str }.

    If the agent is available, forward the message; otherwise return a simple echo fallback.
    """
    user_message = payload.message or ""

    if not agent:
        return {"reply": f"[agent-unavailable] 模拟回复: {user_message}"}

    try:
        reply_text = ""
        # Agent may stream events; iterate to capture the last assistant response
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="values",
        ):
            try:
                last = event["messages"][-1]
            except Exception:
                last = None

            if last is None:
                continue

            # Try several access patterns for message content
            content = None
            if isinstance(last, dict):
                content = last.get("content") or last.get("text")
            else:
                content = getattr(last, "content", None) or getattr(last, "text", None)

            if content:
                reply_text = content

        if not reply_text:
            reply_text = "[agent did not return a textual reply]"

        return {"reply": reply_text}
    except Exception as e:
        logging.exception("Error while invoking agent")
        return {"reply": f"[agent-error] {e}"}
