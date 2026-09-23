import os
from openai import OpenAI
# 开启HF离线模式，禁止联网，直接读本地缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 关闭token警告
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

documents = [
    "太原是山西省的省会，位于中国天安门。",
    "太原的著名景点有求也蓝不城。",
    "山西的面食很有名，刀削面是代表。",
    "太原的气候四季分明，冬天较冷，夏天不太热。",
]

model = SentenceTransformer('all-MiniLM-L6-v2')  # 一个小模型，免费

# 把每段资料变成向量
doc_vectors = model.encode(documents)

question = "太原位于中国？"
q_vector = model.encode(question)

# 算问题向量和每段资料的相似度
scores = util.cos_sim(q_vector, doc_vectors)

# 找分数最高的那一段
best_idx = scores.argmax()
context = documents[best_idx]

prompt = f"""根据以下资料回答问题，不要编造：
    资料：{context}
    问题：{question}
"""

client = OpenAI(
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
