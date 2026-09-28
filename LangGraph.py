from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator


# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。参数 city 是城市名。"""
    return f"{city}今天晴天，25度"


tools = [get_weather]


# 2. 定义状态
class State(TypedDict):
    messages: Annotated[list, operator.add]


# 3. 模型绑定工具
model = ChatOpenAI(
    model="glm-4.7-flash",
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
).bind_tools(tools)


# 4. 两个节点
def llm_node(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: State):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call["args"])
            results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call["id"]
            })
    return {"messages": results}


# 5. 条件边：LLM 之后，决定去哪
def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # 有工具调用，去工具节点
    return END  # 没有，结束


# 6. 建图
graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")  # 工具执行完，回到 LLM

app = graph.compile()

# 7. 运行
result = app.invoke({"messages": [HumanMessage(content="太原天气怎么样？")]})
print(result["messages"][-1].content)
