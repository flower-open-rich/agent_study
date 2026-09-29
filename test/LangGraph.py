import os
import time

import requests

import streamlit as st
# HF离线模式，直接读本地缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import TypedDict, Annotated
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# 替换成Sqlite持久化
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

import streamlit as st

# ========== 1. RAG 资源 ==========
model_embed = SentenceTransformer('BAAI/bge-small-zh-v1.5')
index = faiss.read_index("my_index.faiss")
with open("../my-agent/my_docs.txt", encoding="utf-8") as f:
    documents = [p.strip() for p in f.read().split("\n\n") if p.strip()]


# ========== 2. 工具 ==========
@tool
def search_my_docs(query: str) -> str:
    """搜索我的资料库。当用户问关于太原、山西的问题时使用。"""
    t0 = time.time()
    q_vector = model_embed.encode(query).astype('float32')
    _, indices = index.search(np.array([q_vector]), 3)
    res = "\n".join(documents[i] for i in indices[0])
    t1 = time.time()
    print(f"【RAG检索总耗时】{t1 - t0:.2f}s")
    return res


@tool
def get_weather(city, **kwargs):
    for attempt in range(3):
        try:
            url = f"https://wttr.in/{city}?format=3"
            response = requests.get(url, timeout=10)
            return response.text
        except Exception as e:
            if attempt == 2:  # 最后一次还失败
                return f"查天气失败：{e}"
            time.sleep(1)


tools = [search_my_docs, get_weather]


# ========== 3. 状态与模型 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatOpenAI(
    model="glm-4-flash",
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
).bind_tools(tools)


def llm_node(state: State):
    t0 = time.time()
    resp = model.invoke(state["messages"])
    t1 = time.time()
    print(f"【LLM API耗时】{t1 - t0:.2f}s")
    return {"messages": [resp]}


def should_continue(state: State):
    return "tools" if state["messages"][-1].tool_calls else END


# ========== 4. 建图 + SQLite记忆 ==========
graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

# 打开sqlite数据库，文件名叫 chat_memory.db
conn_string = "chat_memory.db"
with SqliteSaver.from_conn_string(conn_string) as memory:
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "user_1"}}

    # ========== 5. 运行 ==========
    print("输入「退出」结束对话")
    while True:
        question = input("你：").strip()
        if question in ("退出", "exit", "quit"):
            break
        if not question:
            continue
        result = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        print("Agent：", result["messages"][-1].content)
