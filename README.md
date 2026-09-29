# 项目说明

项目分为两个目录：

- `test`：Agent 学习过程代码（循序渐进，用于学习验证）
- `my-agent`：整合完成的最终成品项目

## 📂 test（学习实验目录）

> 
> 从零上手大模型 Agent、RAG、LangGraph 等技术，分阶段迭代验证所有核心能力

1. **hand-write-agent.py**｜手写简易 Agent
   - 首次调用 LLM，使用智谱 GLM-4-Flash（免费模型）
   - 核心目标：跑通基础交互，实现 “输入一句话，返回一句话”
   - 阶段成果：基础对话，支持输出`{}`JSON 格式内容
   - 进阶迭代：让 LLM 输出 JSON 做决策、增加循环实现自主多轮思考、处理 LLM 输出格式不稳定问题、接入真实 API、实现多轮对话、第一次连接 GitHub
2. **OpenAI Agents SDK.py**｜OpenAI Agents SDK 框架实践
   - 使用框架跑通最小 Demo
   - 集成自定义工具
   - 框架自带多轮对话 + 对话持久化记忆
3. **RAG_test.py**｜最小手写 RAG 原型
   - 搭建最简 RAG 完整架构
   - 实现文件读取、文本切块、相似度检索
   - 基于 FAISS 构建向量库
   - 适配中文向量模型 `BAAI/bge-small-zh-v1.5`
4. **build_index.py**｜RAG 索引构建模块
   - 单独抽离索引构建逻辑，**仅执行一次**，预生成向量库
5. **query.py**｜RAG 检索查询模块
   - 读取已构建好的向量索引，执行用户问题检索与问答
6. **LangChain.py**｜LangChain 基础案例
   - LangChain 最小案例落地
7. **LangGraph.py**｜LangGraph 智能 Agent 图编排
   - LangGraph 基础图结构搭建
   - LangGraph + RAG + 工具融合
   - 多轮对话 + 智能工具分发
8. **app.py**｜Streamlit 网页聊天界面
   - 基于 Streamlit 实现浏览器可视化聊天页面

## 📂 my-agent（成品项目目录）

整合`test`中验证通过的全部技术，封装为完整可运行的 Agent 应用

- `agent.py`：Agent 核心逻辑
- `app.py`：Streamlit 网页应用入口
- `build_index.py`：RAG 向量索引构建脚本
- `data`目录：存放文档、知识库等资源文件
- `chat_memory.db`：对话持久化数据库

## 其他文件说明

- `requirements.txt`：项目依赖清单
- `README.md`：项目介绍文档
- `*.db / *.db-shm / *.db-wal`：sqlite 对话持久化数据库相关文件