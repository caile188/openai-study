#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/5 12:30
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_community.tools import WriteFileTool
from langchain_community.tools import ReadFileTool
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_experimental.autonomous_agents import AutoGPT
from langchain_openai import ChatOpenAI

# 工具集（Tools）配置
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool()
]

# 记忆系统（Memory）
# OpenAI Embedding 模型
embeddings_model = OpenAIEmbeddings()
# OpenAI Embedding 向量维度
embedding_size = 1536
# 使用Faiss 的IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# 智能体（Agent）初始化，通过 AutoGPT 类整合 LLM、工具和记忆
agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(model_name="gpt-4", temperature=0, verbose=True),
    memory=vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}
    ),
)
# 打印 Auto-GPT 内部的 chain 日志
agent.chain.verbose = True

# 任务执行
agent.run(["2023年成都大运会，中国金牌数是多少"])

# agent.run(["现任美国总统是谁？他今年多大？"])
