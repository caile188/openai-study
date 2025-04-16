#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/4/7 16:25
import functools
import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 定义工具

# Tavily搜索工具，用于搜索
tavily_tool = TavilySearchResults(max_reults=5)


@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
):
    """
    Python REPL 工具，用于执行python代码
    :param code:
    :return:
    """
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"

    return f"Successfully executed:\n```python\n{code}\n```\n"


def agent_node(state, agent, name):
    """
    辅助函数：定义智能体节点函数
    然后使用它分别定义两个智能体节点：Researcher、 Chart_Generator
    :param state: 状态
    :param agent: 智能体
    :param name: 智能体名称
    :return:
    """
    # 修正名称格式，移除空格并确保只包含合法字符
    name = name.replace(" ", "_").replace("-", "_")

    # 调用智能体，获取结果
    result = agent.invoke(state)
    # print(result)

    # 将智能体的输出转换为适合追加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass
    else:
        # 将结果转换为 AIMessage，并排除部分字段
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        "sender": name
    }


def create_agent(llm, tools, tool_message: str, custom_notice: str=""):
    """
    创建智能体（通过为该智能体提供系统消息和可以使用的工具来指定其行为）
    这些智能体将成为图中的节点。
    :param llm:
    :param tools:
    :param tool_message:
    :param custom_notice:
    :return:
    """

    # 定义智能体提示模版，包含系统消息和工具消息
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                "\n{custom_notice}\n"
                " You have access to the following tools: {tool_names}.\n{tool_message}\n\n",
            ),
            MessagesPlaceholder(variable_name="messages"),  # 用于替换的消息占位符

        ]
    )

    # 将系统消息部分和工具名称插入到提示模板中
    prompt = prompt.partial(tool_message=tool_message, custom_notice=custom_notice)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    # 将提示模板与语言模型和工具绑定
    return prompt | llm.bind_tools(tools)


# 定义搜索智能体及其节点
research_llm = ChatOpenAI(model="gpt-4o-mini")
research_agent = create_agent(
    research_llm,
    [tavily_tool],
    tool_message=
        "Before using the search engine, carefully think through and clarify the query.\n"
        "Then, conduct a single search that addresses all aspects of the query in one go",
    custom_notice=(
        "Notice:\n"
        "Only gather and organize information. Do not generate code or give final conclusions, leave that for other assistants."
    ),
)
# 使用 functools.partial 创建研究智能体的节点，指定该节点的名称为 "Researcher"
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# 定义图表生成器智能体及其节点
chart_llm = ChatOpenAI(model="gpt-4o-mini")
chart_agent = create_agent(
    chart_llm,
    [python_repl],
    tool_message=(
        "Create clear and user-friendly charts based on the provided data. "
        "ALWAYS use non-interactive backend by adding:\n"
        "```python\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "```\n"
        "at the beginning. Save the plot to 'gdp_plot.png' using plt.savefig()."
    ),  # 系统消息，指导智能体如何生成图表
    custom_notice="Notice:\n"
    "If you have completed all tasks, respond with FINAL ANSWER.",

)

# 使用 functools.partial 创建图表生成器智能体的节点，指定该节点的名称为 "Chart_Generator"
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart_Generator")

# 创建工具节点，负责工具调用
tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    """
    建立智能体节点间通信 AgentState
    """
    # messages 是传递的消息，使用 Annotated 和 Sequence 来标记类型
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # sender 是发送消息的智能体
    sender: str


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """
    路由器函数，用于决定下一步是执行工具还是结束任务
    :param state:
    :return:
    """
    messages = state["messages"]  # 获取当前状态中的消息列表
    last_message = messages[-1]

    # 如果最新消息包含工具调用，则返回 "call_tool"，指示执行工具
    if last_message.tool_calls:
        return "call_tool"

    # 如果最新消息中包含 "FINAL ANSWER"，表示任务已完成，返回 "__end__" 结束工作流
    if "FINAL ANSWER" in last_message.content:
        return "__end__"

    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"


# 定义工作流
workflow = StateGraph(AgentState)

# 将研究智能体节点、图表生成器智能体节点和工具节点添加到状态图中
workflow.add_node("Researcher", research_node)
workflow.add_node("Chart_Generator", chart_node)
workflow.add_node("call_tool", tool_node)

# 添加开始节点，将流程从 START 节点连接到 Researcher 节点
workflow.add_edge(START, "Researcher")

# 为 "Researcher" 智能体节点添加条件边，根据 router 函数的返回值进行分支
workflow.add_conditional_edges(
    "Researcher",
    router,  # 路由器函数决定下一步
    {
        "continue": "Chart_Generator",  # 如果 router 返回 "continue"，则传递到 Chart_Generator
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "Chart_Generator" 智能体节点添加条件边
workflow.add_conditional_edges(
    "Chart_Generator",
    router,  # 同样使用 router 函数决定下一步
    {
        "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
        "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
        "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
    },
)

# 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
# 工具调用节点不更新 sender 字段，这意味着边将返回给调用工具的智能体
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
    {
        "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
        "Chart_Generator": "Chart_Generator",  # 如果 sender 是 Chart_Generator，则返回给 Chart_Generator
    },
)

# 编译状态图以便后续使用
graph = workflow.compile()

# 执行
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Obtain the GDP of the United States from 2000 to 2020, "
            "and then plot a line chart with Python. End the task after generating the chart。"
            )
        ],
    },
    # 设置最大递归限制
    {"recursion_limit": 20},
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()  # 打印消息内容



