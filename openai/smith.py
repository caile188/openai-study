#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/8 17:09
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

openai_client = wrap_openai(OpenAI())

@traceable(run_type="retriever")
def retriever(query):
    results = ["Harrison worked at Kensho"]
    return results

@traceable
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:

    {docs}""".format(docs="\n".join(docs))

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )


print(rag("where did harrison work"))
