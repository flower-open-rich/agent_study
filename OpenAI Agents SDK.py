import re

import requests
from agents import Agent, Runner, set_default_openai_client, SQLiteSession
from openai import OpenAI, AsyncOpenAI
from agents import function_tool

# 去掉[non-fatal] Tracing client error 401. Response data is redacted.
# 这是Tracing 是 Agents SDK 自带的追踪功能，它想把运行过程上传到 OpenAI 的服务器做记录。但你用的是智谱的 Key，OpenAI 那边不认，所以报 401（没权限）。
from agents import set_tracing_disabled

set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/v1"
)
set_default_openai_client(client)


@function_tool
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


@function_tool
def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression 是算式，比如 123*456 或 (1+2)*3。"""
    # 只允许数字、空格、和 + - * / ( ) 这些符号
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "算式不合法，只支持数字和 + - * / ( )"

    try:
        result = eval(expression)   # 已经检查过，安全了
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


agent = Agent(
    name="Assistant",
    instructions="你是一个有用的助手，可以查天气，计算。",
    model="glm-4-flash",
    tools=[get_weather, calculator],
)

session = SQLiteSession("user_1")

while True:
    user_input = input("你：")
    if user_input == "退出":
        break
    result = Runner.run_sync(agent, user_input, session=session)
    print("Agent:", result.final_output)