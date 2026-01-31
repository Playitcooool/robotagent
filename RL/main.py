from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import sum_two_num, product_twon_num
from langchain.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant, please select proper tools to meet user's request"
)

tools = [sum_two_num, product_twon_num]
chat = ChatOpenAI(
    base_url="http://localhost:1234/v1", model="qwen3-1.7b-mlx", api_key="no_need"
)

agent = create_agent(model=chat, tools=tools, system_prompt=SYSTEM_PROMPT)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我计算一下1与2的和"}]}
)

messages = response["messages"]

trajectory = []

for msg in response["messages"]:
    # 只关心这三类
    if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
        item = {
            "role": (
                "human"
                if isinstance(msg, HumanMessage)
                else "ai" if isinstance(msg, AIMessage) else "tool"
            ),
            "type": msg.__class__.__name__,
            "content": msg.content,
        }

        # ToolMessage 特有信息（非常重要）
        if isinstance(msg, ToolMessage):
            item["tool_name"] = msg.name
            item["tool_call_id"] = msg.tool_call_id

        trajectory.append(item)


import json

with open("trajectory.json", "w", encoding="utf-8") as f:
    json.dump(trajectory, f, ensure_ascii=False, indent=2)
