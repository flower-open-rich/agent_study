import os
os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 读索引
index = faiss.read_index("my_index.faiss")

# 读原文（顺序必须跟建索引时一致）
with open("my_docs.txt", "r", encoding="utf-8") as f:
    content = f.read()
documents = [p.strip() for p in content.split("\n\n") if p.strip()]

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 问题
question = input("你：")

# 检索
q_vector = model.encode(question).astype('float32')
distances, indices = index.search(np.array([q_vector]), 3)
# 看最相关的那段，距离是不是太大
if distances[0][0] > 1.5:
    print("抱歉，我的资料里没有相关内容。")
    exit()

contexts = [documents[i] for i in indices[0]]
context = "\n".join(contexts)

# 调 LLM
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