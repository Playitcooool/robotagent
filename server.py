from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import json
import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import (
    SummarizationMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
import tools.AnalysisTool, tools.GeneralTool, tools.SubAgentTool
import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from prompts import MainAgentPrompt, LLMToolSelectorPrompt

# ========== 1. 日志配置（确保输出到Uvicorn控制台） ==========
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# ========== 2. 全局变量定义（关键：提前声明active_agent） ==========
active_agent = None  # 全局agent，启动事件中初始化
with open("config/config.yml", "r", encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
# ========== 3. MCP客户端修正（args应为列表） ==========
client = MultiServerMCPClient(
    {"pybullet": {"transport": "http", "url": "http://localhost:8001/mcp"}}
)


# ========== 4. 加载工具函数（保留并添加日志） ==========
async def get_tools():
    try:
        general_tools = []
        subagent_tools = []
        for func_name in tools.GeneralTool.__all__:
            function = getattr(tools.GeneralTool, func_name)
            general_tools.append(function)
        for func_name in tools.SubAgentTool.__all__:
            function = getattr(tools.SubAgentTool, func_name)
            subagent_tools.append(function)

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
)

# ========== 6. FastAPI应用初始化 ==========
app = FastAPI()


# ========== 7. 启动事件（正确初始化agent） ==========
@app.on_event("startup")
async def startup_event():
    global active_agent  # 关联全局变量
    try:
        # 加载所有工具
        all_tools = await get_tools()
        if not all_tools:
            logger.warning("未加载到任何工具，agent将使用空工具列表")

        # 创建带工具的agent（核心修正）
        active_agent = create_agent(
            model=chatBot,
            tools=all_tools,
            system_prompt=MainAgentPrompt,
            middleware=[
                SummarizationMiddleware(
                    model=config["llm"]["summary"],
                    trigger=("tokens", 4000),
                    messages_to_keep=("message", 20),
                ),
                LLMToolSelectorMiddleware(
                    model=config["llm"]["tool_selector"],
                    max_tools=3,
                    system_prompt=LLMToolSelectorPrompt.SYSTEM_PROMPT,
                ),
                ToolRetryMiddleware(
                    max_retries=3,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
            ],
            checkpointer=InMemorySaver(),
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


@app.post("/api/chat/send")
async def chat_send(payload: ChatIn):
    user_message = payload.message or ""

    # 核心修正：使用全局的active_agent，而非初始空工具的agent
    if not active_agent:
        # 返回流式错误响应（保持格式统一）
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
        )

    def event_stream():
        try:
            # 调用全局active_agent的流式接口
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

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# 启动命令：uvicorn server:app --reload --host 0.0.0.0 --port 8000 --log-level info
