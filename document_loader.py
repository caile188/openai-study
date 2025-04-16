#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/2 19:14

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def loader():
    # docs = TextLoader('./tests/state_of_the_union.txt', encoding='utf-8').load()

    docs = TextLoader('./tests/state_of_the_union.txt', encoding='utf-8').load_and_split()

    print(len(docs))
    # print(type(docs[0]))
    # print(docs[0].page_content[:100])

    # arxiv_docs = ArxivLoader(query='2005.14165', load_max_docs=5).load()
    # print(arxiv_docs[0].metadata)a


def transformer():
    with open('./tests/state_of_the_union.txt', encoding='utf-8') as f:
        state_of_the_union = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True
    )
    # docs = text_splitter.create_documents([state_of_the_union])
    # print(docs[0])

    metadatas = [{"document": 1}, {"document": 2}]
    documents = text_splitter.create_documents([state_of_the_union, state_of_the_union], metadatas=metadatas)
    print(documents[999])


def transformer_html():
    html_text = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>🦜️🔗 LangChain</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                h1 {
                    color: darkblue;
                }
            </style>
        </head>
        <body>
            <div>
                <h1>🦜️🔗 LangChain</h1>
                <p>⚡ Building applications with LLMs through composability ⚡</p>
            </div>
            <div>
                As an open source project in a rapidly developing field, we are extremely open to contributions.
            </div>
        </body>
    </html>
    """
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=60,
        chunk_overlap=0
    )
    html_docs = html_splitter.create_documents([html_text])
    print(html_docs)


def embedding():
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents([
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])

    print(len(embeddings))
    # print(embeddings[0])
    print(len(embeddings[0]))

    embed_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    print(len(embed_query))


def vector_store():
    # 加载长文本
    raw_document = TextLoader('./tests/state_of_the_union.txt', encoding='utf-8').load()
    # 实例化文本分割器
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    # 分割文本
    documents = text_splitter.split_documents(raw_document)

    # 使用 OpenAI 嵌入模型获取分割后文本的嵌入向量，并存储在 Chroma 中
    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings_model)

    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)
    print(docs)

    #551d566581c3662d16d2b2791b0fda4a7085a07f1b9ff09f82f4d954ccac56b7









if __name__ == "__main__":
    # loader()
    # transformer()
    # transformer_html()
    # embedding()
    vector_store()