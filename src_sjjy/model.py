#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 10:56
# @Author   : Raymond Luo
# @File     : model.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from train_config import motif_save_path


class rec_model(nn.Module):
    def __init__(self, node_size, node_dim, neighbors_num):
        super(rec_model, self).__init__()
        densedim = node_dim
        # 构建词向量
        word_vectors = KeyedVectors.load_word2vec_format(motif_save_path, binary=False)  # 节点向量
        weight = torch.randn(node_size + 1, node_dim)
        for index in word_vectors.index2word:
            weight[int(index), :] = torch.from_numpy(word_vectors.get_vector(index))
        self.emb = nn.Embedding.from_pretrained(weight, freeze=False)
        # self.emb = nn.Embedding(node_size, node_dim)
        # initrange = (2.0 / (node_size + node_dim)) ** 0.5  # Xavier init
        # self.emb.weight.data.uniform_(-initrange, initrange)
        # User 用的
        self.functionA = nn.Sequential(
            nn.Linear(31, densedim),
            nn.ReLU(),
            nn.Linear(densedim, densedim),
            nn.ReLU()
        )
        # target 用的
        self.functionT = nn.Sequential(
            nn.Linear(31, densedim),
            nn.ReLU(),
            nn.Linear(densedim, densedim),
            nn.ReLU()
        )
        # Atticion
        self.attition = nn.Sequential(
            nn.Linear(node_dim + 31, node_dim + 31),
            nn.Tanh(),
            nn.Linear(node_dim + 31, 1),
            nn.Softmax(dim=1)
        )

        self.dense = nn.Sequential(
            nn.Linear(neighbors_num * (node_dim + 31), densedim),  # merge 邻居用的
            nn.ReLU(),
            nn.Linear(densedim, densedim),
            nn.ReLU())
        self.dense2 = nn.Sequential(nn.Linear(3 * densedim, densedim),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(densedim))  # 综合
        self.output = nn.Sequential(nn.Linear(densedim * 2, densedim),
                                    nn.Dropout(0.5),
                                    nn.Linear(densedim, 1))  # output

    def apply_attition(self, x):
        att_score = self.attition(x)
        scored_x = x * att_score
        return scored_x

    def get_emb(self, idx):
        '''
        获得用户的uuid
        :param feature: input shape: (batch size, 29)
        :return:
        '''
        return self.emb(idx)  # 返回emb特征

    def merge_neighbor(self, neighbor):
        neighbor_graph_feature = self.emb(
            neighbor[:, :, 0].view((neighbor.size()[0], -1)).long())  # 转成 b * r 输入，获得embeding结果 (b, r, nodedim)
        neighbor_feature = neighbor[:, :, 1:]  # (b, r , 28)
        neighbor_feature_concat = torch.cat((neighbor_graph_feature, neighbor_feature), dim=2)  # (b, r , nodedim+28)
        neighbor_feature_att = self.apply_attition(neighbor_feature_concat)
        neighbor_feature_att = neighbor_feature_att.view(neighbor.size()[0], -1)  # (b, r * (nodedim+28))
        return self.dense(neighbor_feature_att)

    def forward(self, user, target, neighborA, neighborB):
        '''

        :param user: b * 29
        :param target: b * 29
        :param neighborA: b * 10 * 29
        :param neighborB: b * 10 * 29
        :return:
        '''
        user_feature = self.functionA(user[:, 1:])  # (batch, densedim) 去除featurez中的idx
        target_feature = self.functionT(target[:, 1:])  # (batch, densedim)
        user_graph_feature = self.get_emb(user[:, 0].long())  # (b，128) # (batch, nodedim)
        target_graph_feature = self.get_emb(target[:, 0].long())  # (batch, nodedim)

        userA_neighbor_feature = self.merge_neighbor(neighborA)  # (b, densedim)
        userB_neighbor_feature = self.merge_neighbor(neighborB)  # (b, densedim)
        userA_concat = self.dense2(torch.cat((user_feature, user_graph_feature, userA_neighbor_feature), dim=1))
        userB_concat = self.dense2(torch.cat((target_feature, target_graph_feature, userB_neighbor_feature), dim=1))
        output = self.output(torch.cat((userA_concat, userB_concat), dim=1))
        # output = torch.sum(userA_concat * userB_concat, dim=1)
        return output
