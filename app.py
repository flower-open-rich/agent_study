import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
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
from langchain_core.messages import HumanMessage


# ========== 加载 Agent（只做一次） ==========
@st.cache_resource
def load_agent():
    # RAG 资源
    model_embed = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    index = faiss.read_index("my_index.faiss")
    with open("my_docs.txt", encoding="utf-8") as f:
        documents = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    # 工具
    @tool
    def search_my_docs(query: str) -> str:
        """搜索我的资料库。当用户问关于太原、山西的问题时使用。"""
        q_vector = model_embed.encode(query).astype('float32')
        _, indices = index.search(np.array([q_vector]), 3)
        return "\n".join(documents[i] for i in indices[0])

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。参数 city 是城市名。"""
        return f"{city}今天晴天，25度"

    tools = [search_my_docs, get_weather]

    # 状态
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 模型
    model = ChatOpenAI(
        model="glm-4-flash",
        api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    ).bind_tools(tools)

    def llm_node(state: State):
        resp = model.invoke(state["messages"])
        return {"messages": [resp]}

    def should_continue(state: State):
        return "tools" if state["messages"][-1].tool_calls else END

    # 建图
    graph = StateGraph(State)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    # 持久化记忆
    conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = graph.compile(checkpointer=memory)

    return app


# ========== 网页界面 ==========
st.title("我的 Agent")

app = load_agent()

# 网页版的记忆
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 输入框
if prompt := st.chat_input("问点什么"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 调 Agent
    result = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "user_1"}}
    )
    answer = result["messages"][-1].content

    # 显示 Agent 回复
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)