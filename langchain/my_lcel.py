#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/31 16:00

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def demo1():

    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate([
        ("human", "讲个关于{topic}的笑话吧")
    ])
    # print(prompt.invoke({"topic": "程序员"}))
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke({"topic": "程序员"})
    print(result)


if __name__ == "__main__":
    demo1()
