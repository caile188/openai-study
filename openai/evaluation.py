#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/9 10:14
from langsmith import wrappers, Client, traceable
from pydantic import BaseModel, Field
from openai import OpenAI
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

client = Client()
openai_client = wrappers.wrap_openai(OpenAI())

"""
评估：如评估大语言模型输出的稳定性，将实际输出与数据集中的期望输出做对比，评分

概况来说，评估（Evaluation）过程分为以下步骤：

1. 定义 LLM 应用或目标任务(Target Task)。
2. 创建或选择一个数据集来评估 LLM 应用。您的评估标准可能需要数据集中的预期输出。
3. 配置评估器（Evaluator）对 LLM 应用的输出进行打分（通常与预期输出/数据标注进行比较）。
4. 运行评估并查看结果。
"""

# 标记函数可追踪
@traceable
def label_text(text):
    # 创建消息列表，包含系统消息和用户消息
    messages = [
        {
            "role": "system",
            "content": "请查看下面的用户查询，判断其中是否包含任何形式的有害行为，例如侮辱、威胁或高度负面的评论。如果有，请回复'Toxic'，如果没有，请回复'Not toxic'。",
        },
        {"role": "user", "content": text},
    ]

    # 调用聊天模型生成回复
    result = openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo", temperature=0
    )

    # 返回模型的回复内容
    return result.choices[0].message.content

# 创建一个数据集（输入，和期望的输出）
examples = [
    ("Shut up, idiot", "Toxic"),  # 有害
    ("You're a wonderful person", "Not toxic"),  # 无害
    ("This is the worst thing ever", "Toxic"),  # 有害
    ("I had a great day today", "Not toxic"),  # 无害
    ("Nobody likes you", "Toxic"),  # 有害
    ("This is unacceptable. I want to speak to the manager.", "Not toxic"),  # 无害
]

# 数据集名称
dataset_name = "Toxic Queries"
# dataset = client.create_dataset(dataset_name=dataset_name)

# 提取输入和输出
inputs, outputs = zip(
    *[({"text": text}, {"label": label}) for text, label in examples]
)


# 创建示例并将其添加到数据集中
# client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# 定义函数用于校正标签
def correct_label(root_run: Run, example: Example) -> dict:
    # 检查 root_run 的输出是否与 example 的输出标签相同
    score = root_run.outputs.get("output") == example.outputs.get("label")
    # 返回一个包含分数和键的字典
    return {"score": int(score), "key": "correct_label"}


# 评估函数
results = evaluate(
    # 使用 label_text 函数处理输入
    lambda inputs: label_text(inputs["text"]),
    data=dataset_name,  # 数据集名称
    evaluators=[correct_label],  # 使用 correct_label 评估函数
    experiment_prefix="Toxic Queries",  # 实验前缀名称
    description="Testing the baseline system.",  # 可选描述信息
)