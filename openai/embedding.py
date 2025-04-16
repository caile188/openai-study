#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/17 11:48

from openai import OpenAI
import pandas as pd
import tiktoken
import ast
# 导入 NumPy 包，NumPy 是 Python 的一个开源数值计算扩展。这种工具可用来存储和处理大型矩阵，
# 比 Python 自身的嵌套列表（nested list structure)结构要高效的多。
import numpy as np
# 从 matplotlib 包中导入 pyplot 子库，并将其别名设置为 plt。
# matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
import matplotlib

# 从 sklearn.manifold 模块中导入 TSNE 类。
# TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。
# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.manifold import TSNE

client = OpenAI()


def embedding_text(text, model="text-embedding-3-small"):
    """
    调用 OpenAI Embedding API,获取输入数据的向量表示
    :param text:
    :param model:
    :return:
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )

    print(len(response.data[0].embedding))


def load_data():
    """
    加载数据，并调用 OpenAI Embedding API,获取数据的向量表示，写入文件
    :return:
    """

    # 加载数据集
    input_datapath = "../tests/fine_food_reviews_1k.csv"
    df = pd.read_csv(input_datapath, index_col=0)
    df = df.dropna()
    # 将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
    df['combined'] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    )

    top_n = 1000
    max_tokens = 8000
    df = df.sort_values("Time").tail(top_n)
    df.drop("Time", axis=1, inplace=True)

    encoding = tiktoken.get_encoding("cl100k_base")
    # 计算每条评论的token数量。我们通过使用encoding.encode方法获取每条评论的token数，然后把结果存储在新的'n_tokens'列中。
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

    # 如果评论的token数量超过最大允许的token数量，我们将忽略（删除）该评论。
    # 我们使用.tail方法获取token数量在允许范围内的最后top_n（1000）条评论。
    df = df[df.n_tokens <= max_tokens]

    #逐行调用 OpenAI Embedding API
    # df["embedding"] = df.combined.apply(embedding_text)
    # output_datapath = "data/fine_food_reviews_with_embeddings_1k_1126.csv"
    # df.to_csv(output_datapath)

# 读取 fine_food_reviews_with_embeddings_1k 嵌入文件
embedding_datapath = "../tests/fine_food_reviews_with_embeddings_1k.csv"
df_embedded = pd.read_csv(embedding_datapath, index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)
# print(len(matrix))
# 使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)
# print(vis_dims)
# 定义了五种不同的颜色，用于在可视化中表示不同的等级
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
# 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
# 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
color_indices = df_embedded.Score.values - 1
# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)
# 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

# 为图形添加标题
plt.title("Amazon ratings visualized in language using t-SNE")