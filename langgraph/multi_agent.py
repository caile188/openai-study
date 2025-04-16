#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/4/16 10:32

from typing import Annotated, Sequence, TypedDict, Literal
from operator import add
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


class AppConfig:
    """
    配置类（集中管理配置参数）
    """
    MODEL_NAME = "gpt-4o"
    # MODEL_NAME = "deepseek-chat"
    TAVILY_MAX_RESULTS = 5
    PLOT_FILENAME = "gdp_plot.png"
    RECURSION_LIMIT = 20


class AgentState(TypedDict):
    """
    定义智能体节点间通信的 State
    """
    messages: Annotated[Sequence[BaseMessage], add]
    sender: str


class ToolManager:
    """
    tools 管理
    """

    @staticmethod
    def get_tools():
        """初始化并返回所有工具"""
        return [
            TavilySearchResults(max_results=AppConfig.TAVILY_MAX_RESULTS),
            ToolManager.python_repl
        ]

    @staticmethod
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
            return f"执行失败. Error: {repr(e)}"

        return f"成功执行:\n```python\n{code}\n```\n"


class AgentFactory:
    """
    Agent 管理
    """
    @staticmethod
    def create_agent(llm, tools, tool_message: str, custom_notice: str = ""):
        """
        创建智能体的工厂方法
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个协作型AI助手，与其他助手共同工作。"
             "使用工具逐步解决问题，若无法完成则转交后续处理。"
             "当获得最终答案时，必须用【FINAL ANSWER】开头标注。\n"
             f"{custom_notice}\n"
             f"可用工具：{', '.join(t.name for t in tools)}\n"
             f"{tool_message}"
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | llm.bind_tools(tools)

    @staticmethod
    def create_agent_node(agent, name: str):
        """创建智能体节点"""

        def _node(state):
            result = agent.invoke(state)
            message = AIMessage(
                content=result.content,
                additional_kwargs=result.additional_kwargs,
                name=name.replace(" ", "_")
            )
            return {"messages": [message], "sender": name}

        return _node

    @staticmethod
    def create_research_agent():
        """创建研究型智能体"""
        return AgentFactory.create_agent(
            llm=ChatOpenAI(model=AppConfig.MODEL_NAME),
            # llm=ChatDeepSeek(model=AppConfig.MODEL_NAME),
            tools=[ToolManager.get_tools()[0]],
            tool_message=(
                "使用搜索工具前，请先明确查询需求，确保单次搜索能覆盖所有相关方面。"
            ),
            custom_notice=(
                "特别注意：\n"
                "- 仅执行信息采集与结构化整理\n"
                "- 禁止生成代码或最终结论\n"
                "- 后续处理交由其他专业助手完成"
            )
        )

    @staticmethod
    def create_chart_agent():
        """创建图表生成智能体"""
        return AgentFactory.create_agent(
            llm=ChatOpenAI(model=AppConfig.MODEL_NAME),
            # llm=ChatDeepSeek(model=AppConfig.MODEL_NAME),
            tools=[ToolManager.get_tools()[1]],
            tool_message=(
                "基于提供的数据创建清晰且用户友好的图表。"
                "请始终通过添加以下代码使用非交互式后端并添加字体配置：\n"
                "```python\n"
                "import matplotlib\n"
                "matplotlib.use('Agg')"
                "```\n"
                "并置于代码开头。使用 plt.savefig() 将图表保存至 'gdp_plot.png' 文件。"
            ),
            custom_notice="图表工程师：负责数据可视化"
        )


class WorkflowBuilder:
    """
    创建工作流
    """
    def __init__(self):
        self.workflow = StateGraph(AgentState)

    def build(self):
        """构建完整工作流, 并编译"""
        self._add_nodes()
        self._setup_edges()
        return self.workflow.compile()

    def _add_nodes(self):
        """添加所有节点"""
        research_agent = AgentFactory.create_research_agent()
        chart_agent = AgentFactory.create_chart_agent()

        self.workflow.add_node("Researcher", AgentFactory.create_agent_node(research_agent, "Researcher"))
        self.workflow.add_node("Chart_Generator", AgentFactory.create_agent_node(chart_agent, "Chart_Generator"))
        self.workflow.add_node("call_tool", ToolNode(ToolManager.get_tools()))

    def _setup_edges(self):
        """添加边，配置节点关系"""
        self.workflow.add_edge(START, "Researcher")
        self.workflow.add_conditional_edges(
            "Researcher",
            self._router,  # 路由器函数决定下一步
            {
                "continue": "Chart_Generator",  # 如果 router 返回 "continue"，则传递到 Chart_Generator
                "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
                "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
            },
        )
        self.workflow.add_conditional_edges(
            "Chart_Generator",
            self._router,  # 同样使用 router 函数决定下一步
            {
                "continue": "Researcher",  # 如果 router 返回 "continue"，则回到 Researcher
                "call_tool": "call_tool",  # 如果 router 返回 "call_tool"，则调用工具
                "__end__": END  # 如果 router 返回 "__end__"，则结束工作流
            },
        )

        # 为 "call_tool" 工具节点添加条件边，基于“sender”字段决定下一个节点
        self.workflow.add_conditional_edges(
            "call_tool",
            lambda x: x["sender"],  # 根据 sender 字段判断调用工具的是哪个智能体
            {
                "Researcher": "Researcher",
                "Chart_Generator": "Chart_Generator",
            },
        )

    def _router(self, state) -> Literal["call_tool", "__end__", "continue"]:
        """路由决策逻辑"""
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


if __name__ == "__main__":
    # 初始化工作流
    workflow = WorkflowBuilder().build()

    # 执行工作流
    inputs = {
        "messages": [
            HumanMessage(content=(
                "获取美国2000-2020年GDP数据。"
                "使用Python生成折线图。"
                "图表中的信息使用英文显示，"
                "图表生成后结束任务。"
            ))
        ]
    }

    for event in workflow.stream(inputs, {"recursion_limit": AppConfig.RECURSION_LIMIT}, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
