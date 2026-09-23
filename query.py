import os
os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 读索引和原文（只读一次）
index = faiss.read_index("my_index.faiss")

with open("my_docs.txt", "r", encoding="utf-8") as f:
    content = f.read()
documents = [p.strip() for p in content.split("\n\n") if p.strip()]

# 加载模型（只加载一次）
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

client = OpenAI(
    api_key="ad0e34a9b3b4486db1cac65af204ccbc.Y2FcPICz0Y1h48je",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 多轮问答
while True:
    question = input("你：")
    if question == "退出":
        break

    # 检索
    q_vector = model.encode(question).astype('float32')
    distances, indices = index.search(np.array([q_vector]), 3)
    # 距离，阈值
    if distances[0][0] > 0.9:
        print("Agent：抱歉，我的资料里没有相关内容。")
        continue

    print("最相关距离：", distances[0][0])

    contexts = [documents[i] for i in indices[0]]
    # for i, c in enumerate(contexts):
    #     print(f"  [{i}] {c}")
    context = "\n".join(contexts)

    # 调 LLM
    prompt = f"""根据以下资料回答问题，不要编造：
                    资料：{context}
                    问题：{question}
    """

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )

    print("Agent：", response.choices[0].message.content)
