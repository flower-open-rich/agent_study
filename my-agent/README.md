# 我的 RAG Agent

一个基于 LangGraph 的智能问答 Agent，能查资料、查天气、记住对话。

## 功能

- **RAG 检索**：基于 FAISS + 中文向量模型，从资料库找答案
- **工具调用**：Agent 自己决定要不要调工具
- **多轮对话**：基于 LangGraph Checkpointer 持久化记忆
- **网页界面**：Streamlit

## 技术栈

- LangGraph（Agent 编排）
- LangChain（工具定义）
- FAISS + BAAI/bge-small-zh-v1.5（向量检索）
- Streamlit（界面）
- 智谱 GLM-4（LLM）

## 架构

```
用户提问
  ↓
LangGraph Agent
  ↓
判断是否需要工具
  ├─ 是 → 调 search_my_docs / get_weather → 回 LLM
  └─ 否 → 直接回答
```

## 项目结构

```
my-agent/
├── app.py              # Streamlit 界面
├── agent.py            # Agent 核心逻辑
├── build_index.py      # 建索引脚本
├── requirements.txt    # 依赖
├── data/
│   └── my_notes.txt    # 资料原文
└── .gitignore
```

## 怎么跑

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 设置 API Key

   在项目根目录建 `.env` 文件：
   ```
   ZHIPU_API_KEY=你的智谱Key
   ```

3. 建索引
   ```bash
   python build_index.py
   ```

4. 启动
   ```bash
   streamlit run app.py
   ```

5. 修改知识库

   资料文件在 `data/my_notes.txt`，修改后需重新运行 `build_index.py` 重建索引。