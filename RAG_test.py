import os
from openai import OpenAI
# 开启HF离线模式，禁止联网，直接读本地缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 关闭token警告
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

with open("my_notes.txt", "r", encoding="utf-8") as f:
    content = f.read()
documents = [p.strip() for p in content.split("\n\n") if p.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')  # 一个小模型，免费

# 把每段资料变成向量
doc_vectors = model.encode(documents)

question = "中北大学有什么好玩的"
q_vector = model.encode(question)

# 算问题向量和每段资料的相似度
scores = util.cos_sim(q_vector, doc_vectors)

# 找分数最高的前3段
top_k = 3
top_indices = scores.argsort(descending=True)[0][:top_k]
contexts = [documents[i] for i in top_indices]
context = "\n".join(contexts)

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
