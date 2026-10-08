import os
from dotenv import load_dotenv

# 导入api
load_dotenv()
api_key = os.environ.get("ZHIPU_API_KEY")
import requests

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 加载 RAG 资源
model_embed = SentenceTransformer('BAAI/bge-small-zh-v1.5')
index = faiss.read_index("my-agent/my_index.faiss")
with open("my-agent/my_docs.txt", encoding="utf-8") as f:
    documents = [p.strip() for p in f.read().split("\n\n") if p.strip()]


@tool
def search_my_docs(query: str) -> str:
    """搜索我的资料库。当用户问关于太原、山西的问题时使用。"""
    q_vector = model_embed.encode(query).astype('float32')
    _, indices = index.search(np.array([q_vector]), 3)
    return "\n".join(documents[i] for i in indices[0])


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。参数 city 是城市名，比如 太原。"""
    for attempt in range(3):
        try:
            url = f"https://wttr.in/{city}?format=3"
            response = requests.get(url, timeout=10)
            return response.text
        except Exception as e:
            if attempt == 2:
                return f"查天气失败：{e}"


tools = [search_my_docs, get_weather]


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ.get("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
).bind_tools(tools)


def llm_node(state: State):
    resp = model.invoke(state["messages"])
    return {"messages": [resp]}


def should_continue(state: State):
    return "tools" if state["messages"][-1].tool_calls else END


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)
