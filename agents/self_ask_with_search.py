#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/4 10:25
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

llm = ChatOpenAI(model="gpt-4", temperature=0)
#实例化查询工具
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    )
]

#实例化 SELF_ASK_WITH_SEARCH Agent
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)

# self_ask_with_search.invoke("成都举办的大运会是第几届大运会？2023年大运会举办地在哪里？")
self_ask_with_search.invoke("现任美国总统是谁？他今年多大了？")