import requests
from openai import OpenAI
import json

client = OpenAI(
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


def get_weather(city, **kwargs):
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"查天气失败：{e}"


def none():
    return "不需要工具"


def calculator(expression, **kwargs):
    return str(eval(expression))     #eval 非常危险！如果传入用户可控的恶意字符串，可以执行任意代码。


tools = {
    "get_weather": {
        "function": get_weather,
        "description": "查天气，参数 city（城市名）"
    },
    "calculator": {
        "function": calculator,
        "description": "算数学，参数 expression（算式）"
    },
    "none": {
        "function": none,
        "description": "不需要工具"
    },
}


tool_text = ""
for name, info in tools.items():
    tool_text += f"- {name}：{info['description']}\n"

system_content = f"""
    {tool_text}
    先跟用户打个招呼，再输出 JSON。
    每次只使用一个工具，输出一个json，不要输出其他任何文字。
    格式：{{"tool": "工具名", "args": {{...}}, "done": true或false, "answer": "回答"}}。
    规则：如果要调用工具，done 设为 false，answer 设为空字符串；如果不需要调用工具、可以直接回答用户，done 设为 true，answer 填你的回答。
"""

messages = [
     {"role": "system", "content": system_content},
     {"role": "user", "content": "太原天气怎么样"}
]


for i in range(10):
    print(f"--- 第 {i + 1} 圈 ---")
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )
    text = response.choices[0].message.content
    print("LLM 说：", text)
    # 清洗：只留 { 到 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    # 解析，失败就跳过这圈
    try:
        decision = json.loads(text)
    except Exception as e:
        print("JSON 解析失败：", e)
        break
    messages.append({"role": "assistant", "content": text})
    done = decision.get('done', False)
    if done:
        print(decision.get('answer', '（模型没给回答）'))
        break
    tool_name = decision['tool']
    tool_info = tools.get(tool_name)
    args = decision['args']
    if tool_info:
        try:
            result = tool_info["function"](**args)
        except Exception as e:
            result = f"工具执行出错：{e}"
    else:
        result = '未知工具'

    messages.append({"role": "user", "content": f"工具调用结果是{result}。请根据这个结果回答用户，不要重复调用同一个工具。"})

