from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict


# 1. 定义状态：图里流转的数据
class State(TypedDict):
    question: str
    answer: str


# 2. 定义节点：一个函数，接收状态，返回更新
model = ChatOpenAI(
    model="glm-4-flash",
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


def answer_node(state: State):
    response = model.invoke(state["question"])
    return {"answer": response.content}


# 3. 建图
graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)

app = graph.compile()

# 4. 运行
result = app.invoke({"question": "用一句话解释什么是Agent"})
print(result["answer"])
