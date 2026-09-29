import os
os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 读文件
with open("my_notes.txt", "r", encoding="utf-8") as f:
    content = f.read()

documents = [p.strip() for p in content.split("\n\n") if p.strip()]

# 算向量
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
doc_vectors = model.encode(documents).astype('float32')

# 建索引
index = faiss.IndexFlatL2(512)
index.add(doc_vectors)

# 存索引
faiss.write_index(index, "my_index.faiss")

# 存原文（关键！）
with open("../my-agent/my_docs.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(documents))

print(f"索引建好，共 {len(documents)} 段资料")