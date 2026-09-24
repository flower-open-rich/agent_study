import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 用智谱（OpenAI 兼容接口）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

def get_weather(city: str) -> str:
    """查询指定城市的天气。参数 city 是城市名。"""
    return f"{city}今天晴天，25度"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个助手，可以用工具查天气。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "太原天气怎么样？"}]})
print(result["messages"][-1].content)