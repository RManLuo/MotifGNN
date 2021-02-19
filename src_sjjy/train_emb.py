#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/11/1 23:30
# @Author   : Raymond Luo
# @File     : train_emb.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import pickle
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import torch.nn as nn
import torch


def train_motif_wordemb(path):
    data = pd.read_csv(path)
    walk_a = data['user_neighbor'].values.tolist()
    walk_b = data['target_neighbor'].values.tolist()
    walk_a.extend(walk_b)
    walk = []
    for line in walk_a:
        new_line = line[1:-1].split(", ")
        walk.append(new_line)
    model = Word2Vec(walk, size=128, window=3, min_count=0, sg=1, workers=12, iter=2, compute_loss=True)
    print("Node2vec loss:", model.get_latest_training_loss())
    model.wv.save_word2vec_format("../model/motif_walk.emb")


def change_emb_index(emb_path, uid2idx_path):
    with open(uid2idx_path, "rb") as f:
        uid2idx = pickle.load(f)
    with open(emb_path, "r") as f:
        emb_file = f.readlines()
    head = 1
    new_file = []
    for line in emb_file:
        if head:
            head = 0
            new_file.append(line)
            continue  # 跳过第一行
        line_list = line.split(" ")
        idx = uid2idx[int(line_list[0])]  # uid 2 idx
        line_list[0] = str(idx)  # 转回去
        new_line = " ".join(line_list)
        new_file.append(new_line)
    with open("../model/motif_walk_idx.emb", "w", encoding="utf-8") as f:
        for line in new_file:
            f.write(line)


if __name__ == "__main__":
    # train_motif_wordemb("../data/train_data.csv")
    # change_emb_index("../model/motif_walk.emb", "../data/uid_2_idx.pkl")
    # test
    # 构建词向量
    word_vectors = KeyedVectors.load_word2vec_format("../model/motif_walk_idx.emb", binary=False)  # 节点向量
    weight = torch.FloatTensor(word_vectors.syn0)  # 获取2D numpy矩阵
    emb = nn.Embedding.from_pretrained(weight, freeze=False)
    print(emb(torch.LongTensor([47066])))
