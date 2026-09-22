from agents import Agent, Runner, set_default_openai_client
from openai import OpenAI, AsyncOpenAI

client = AsyncOpenAI(
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/v1"
)

set_default_openai_client(client)

agent = Agent(
    name="Assistant",
    instructions="你是一个有用的助手。",
    model="glm-4.7-flash",
)

result = Runner.run_sync(agent, "用一句话解释什么是Agent")
print(result.final_output)
