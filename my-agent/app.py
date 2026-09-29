import streamlit as st
from langchain_core.messages import HumanMessage
from agent import app

st.title("我的 Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("问点什么"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    result = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "user_1"}}
    )
    answer = result["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)