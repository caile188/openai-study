#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/4 10:54
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

import datetime

chat_model = ChatOpenAI(name="gpt-4", temperature=0)


# 加载 LangChain 内置的 Tools
# tools_list = get_all_tool_names()
# print(tools_list)
tools = load_tools(["serpapi", "llm-math"], llm=chat_model)

# 实例化 ZERO_SHOT_REACT Agent
agent = initialize_agent(tools, chat_model, agents=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.invoke(f"首先你应该明确，当前日期为：{datetime.datetime.now()},那么现任美国总统是谁？他现在年龄的0.43次方是多少?")
agent.invoke("北京今天天气如何")