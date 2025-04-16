#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/10 12:10
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 使用 Document Transformers 模块来处理原始数据
loader = TextLoader('../tests/real_estate_sales_data.txt')
documents = loader.load()
# with open("../tests/real_estate_sales_data.txt") as f:
#     real_estate_sales = f.read()

text_splitter = CharacterTextSplitter(
    separator=r'\d+\.',
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True,
)
docs = text_splitter.split_documents(documents)

# 使用 Faiss 向量数据库，持久化存储房产销售 问答对
db = FAISS.from_documents(docs, OpenAIEmbeddings())
query = "小区吵不吵？"
# answer_list = db.similarity_search(query)
#
# for ans in answer_list:
#     print(ans.page_content + "\n")

# 使用 retriever 从向量数据库中获取结果¶
# 使用参数 k 指定返回结果数量
# topK_retriever = db.as_retriever(search_kwargs={"k": 3})
# topk_docs = topK_retriever.invoke(query)
# for doc in topk_docs:
#     print(doc.page_content + "\n")

query = "我想买别墅，你们有么"
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)
# new_docs = retriever.invoke(query)
# for doc in new_docs:
#     print(doc.page_content + "\n")

# 当向量数据库中没有合适答案时，使用大语言模型的能力
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_prompt = (
    "你是一名房产中介，会收到用户的一些问题"
    "请使用给定的上下文来回答问题。"
    "如果上下文中没有找到答案，就说：对不起，这个问题超出了我的能力范围，稍后会有专业人士主动联系您，请稍等片刻"
    "最多使用三句话，保持答案简洁。"
    "上下文：{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
result = chain.invoke({"input": query})
print(result)





