#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/10 11:01
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# 实例化文档加载器
loader = TextLoader("../tests/state_of_the_union.txt")
# 加载文档
documents = loader.load()

# 实例化文档分割器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# OpenAI Embedding 模型
embeddings = OpenAIEmbeddings()
# FAISS 向量数据库，使用 docs 的向量作为初始化存储
# db = FAISS.from_documents(docs, embeddings)
#
# # 在 Faiss 中进行相似度搜索，找出与 query 最相似结果
query = "What did the president say about Ketanji Brown Jackson"
# result_docs = db.similarity_search(query)
#
# # 持久化存储 Faiss DB
# db.save_local("faiss_index")

# 加载Faiss DB
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
result_docs = new_db.similarity_search(query)
print(result_docs[0].page_content)



