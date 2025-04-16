#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/4/2 16:35
import json
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    """
    定义状态类型，继承自 TypedDict, 并使用 add_messages 函数将消息追加到现有列表
    """
    messages: Annotated[list, add_messages]


def get_tools():
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    return tools


def chatbot(state: State):
    """
    定义聊天机器人的节点函数，接收当前状态并返回更新的消息列表
    :param state:
    :return:
    """
    # print(state)

    chat_model = ChatOpenAI(model="gpt-4o-mini")
    tools = get_tools()
    llm_with_tools = chat_model.bind_tools(tools)

    return {"messages": [llm_with_tools.invoke(state["messages"])]}


class BasicToolNode:
    """
    定义 BasicToolNode，用于执行工具请求
    一个在最后一条 AIMessage 中执行工具请求的节点
    该节点会检查最后一条 AI 消息中的工具调用请求，并依次执行这些工具调用。
    """
    def __init__(self, tools: list) -> None:
        # tools 是一个包含所有可用工具的列表，我们将其转化为字典，
        # 通过工具名称（tool.name）来访问具体的工具
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        执行工具调用
        :param inputs: 包含 "messages" 键的字典，"messages" 是对话消息的列表，其中最后一条消息可能包含工具调用的请求。
        :return: 包含工具调用结果的消息列表
        """
        # 获取消息列表中的最后一条消息，判断是否包含工具调用请求
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("输入中未找到消息")

        outputs = []

        # 遍历工具调用请求，执行工具，并将结果返回
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )

            # 将工具调用结果作为 ToolMessage 保存下来
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": outputs}


def route_tools(state: State):
    """
    定义路由函数，检查工具调用
    使用条件边来检查最后一条消息中是否有工具调用。
    :param state: 状态字典或消息列表，用于存储当前对话的状态和消息
    :return:
    如果最后一条消息包含工具调用，返回 "tools" 节点，表示需要执行工具调用；
    否则返回 "__end__"，表示直接结束流程。
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif message := state.get("messages", []):
        ai_message = message[-1]
    else:
        raise ValueError(f"输入状态中未找到消息：{state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def run():
    # 创建一个状态图对象，传入状态定义
    graph_builder = StateGraph(State)
    # 添加聊天节点
    graph_builder.add_node("chatbot", chatbot)

    # 将BasicToolNode 添加到状态图中
    tools = get_tools()
    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # 定义聊天机器人对话流程（状态图的起点）
    graph_builder.add_edge(START, "chatbot")

    # 添加条件边
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {
            "tools": "tools",
            END: END
        }
    )

    # 当工具调用完成后，返回到聊天机器人节点以继续对话
    graph_builder.add_edge("tools", "chatbot")

    # 编译图可视化, 将MemorySaver 作为检查点传递
    graph = graph_builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    while True:
        # 获取用户输入
        user_input = input("User：")

        # 可以随时通过输入 “quit”、“exit” 或 “q” 退出聊天记录
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # 将每次用户输入的内容传递给 graph.stream，用于聊天机器人状态处理
        # 构建符合类型定义的状态输入
        current_state = {
            "messages": [{
                "role": "user",
                "content": user_input
            }]
        }

        for event in graph.stream(current_state, config):
            # print(event)
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)



def tavily_test():
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    rs = tool.invoke("weather in beijing")
    print(rs)



if __name__ == "__main__":
    run()
    # tavily_test()





