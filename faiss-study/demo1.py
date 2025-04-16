#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/27 15:31

import numpy as np
import faiss


def norm():
    # 生成随机向量（3个样本，4维）
    vectors = np.array([
        [1.2, 3.4, 0.5, 2.3],
        [0.8, 1.1, 2.0, 4.5],
        [2.0, 2.0, 2.0, 2.0]
    ], dtype=np.float64)

    # L2归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    print("原始向量：\n", vectors)
    print("归一化后向量：\n", normalized_vectors)
    print("验证归一化后长度: \n", np.linalg.norm(normalized_vectors, axis=1))


def flat_index():
    # 向量维度
    d = 64
    # 数据量
    nb = 10000

    # 生成随机数据
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')

    # 创建L2距离的Flat索引， 并向索引中添加数据
    index_flat_l2 = faiss.IndexFlatL2(d)
    #print(index_flat_l2.is_trained)
    index_flat_l2.add(xb)

    # 创建内积（IP）的Flat索引，并向索引中添加数据
    index_flat_ip = faiss.IndexFlatIP(d)
    index_flat_ip.add(xb)

    # 查询
    query = np.random.random((1, d)).astype('float32')

    # 返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    D_l2, I_l2 = index_flat_l2.search(query, k=3)
    D_ip, I_ip = index_flat_ip.search(query, k=3)
    print(D_l2)
    print(I_l2)
    print(D_ip)
    print(I_ip)


def ivf_index():
    nlist = 100  # 聚类中心数
    d = 64  # 维度
    nb = 10000  # 数据量

    # 生成随机数据
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')

    # 1. 创建量化器（使用Flat索引）
    quantizer = faiss.IndexFlatL2(d)

    # 2. 创建IVFFlat 索引
    index_ivf_flat = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # 3. 训练索引并添加数据
    index_ivf_flat.train(xb)
    index_ivf_flat.add(xb)

    # 4. 搜索
    index_ivf_flat.nprobe = 10  # 设置搜索的聚类数
    query = np.random.rand(1, d).astype('float32')
    D_ivf, I_ivf = index_ivf_flat.search(query, k=5)
    print(D_ivf)
    print(I_ivf)


def pq_index():
    d = 64
    m = 8  # 子空间数
    nbits = 8  # 每个子空间8位编码
    nlist = 100  # 聚类中心数
    nb = 10000  # 数据量

    # 生成随机数据
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')

    # 1. 创建量化器
    quantizer = faiss.IndexFlatL2(d)

    # 2. 创建IVFPQ索引
    index_ivf_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # 3. 训练与添加数据
    index_ivf_pq.train(xb)
    index_ivf_pq.add(xb)

    # 4. 搜索
    query = np.random.rand(1, d).astype('float32')
    index_ivf_pq.nprobe = 10
    D_pq, I_pq = index_ivf_pq.search(query, k=5)
    print(D_pq)
    print(I_pq)


def hnsw_index():
    M = 32  # 每层图的连接数
    d = 64

    # 创建HNSW索引
    index_hnsw = faiss.IndexHNSWFlat(d, M)
    index_hnsw.hnsw.efConstruction = 100  # 构建时搜索深度
    index_hnsw.hnsw.efSearch = 50  # 查询时搜索深度

    # 向索引中添加数据
    xb = np.random.rand(10000, d).astype('float32')
    index_hnsw.add(xb)

    # 搜索
    query = np.random.rand(1, d).astype('float32')
    D_hnsw, I_hnsw = index_hnsw.search(query, k=5)
    print(D_hnsw, I_hnsw)




if __name__ == "__main__":
    # norm()
    # flat_index()
    # ivf_index()
    # pq_index()
    hnsw_index()
