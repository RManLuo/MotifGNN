#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/8/20 12:32
# @Author   : Raymond Luo
# @File     : recommend_model.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import torch
from torch import nn
from gensim.models import KeyedVectors


class recommend(nn.Module):
    def __init__(self, node_size, node_dim, embed_path):
        super(recommend, self).__init__()
        self.train_mode = True
        # 构建词向量
        word_vectors = KeyedVectors.load_word2vec_format(embed_path, binary=False)  # 节点向量
        weight = torch.FloatTensor(word_vectors.syn0)  # 获取2D numpy矩阵
        self.emb = nn.Embedding.from_pretrained(weight, freeze=False)
        # User 用的
        self.functionA = nn.Sequential(
            nn.Linear(28, 128),
            nn.Linear(128, 128)
        )
        # target和negative用的
        self.functionT = nn.Sequential(
            nn.Linear(28, 128),
            nn.Linear(128, 128)
        )
        # 给用户和邻居用的，输入用户特征+embedding，用户
        self.functionB = nn.Sequential(
            nn.Linear(28, 128),
            nn.Linear(128, 128),
        )

        self.output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        # # 迭代循环初始化参数
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #     # 也可以判断是否为conv2d，使用相应的初始化方式
        #     elif isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight.item(), 1)
        #         nn.init.constant_(m.bias.item(), 0)

    def diff(self, featureA, featureB):
        '''
        计算欧式距离
        :param featureA:
        :param featureB:
        :return:
        '''
        return torch.abs(featureA - featureB)

    def get_index(self, feature):
        '''
        获得用户的uuid
        :param feature: input shape: (batch size, 29)
        :return:
        '''
        return feature[:, 0].long()  # 只返回index

    def train(self, mode=True):
        self.train_mode = mode
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, user, neighbor, target_user, negative_user):
        # 推荐部分，用户特征嵌入高维
        user_feature = self.functionA(user[:, 1:])  # (batch, 128)
        target_feature = self.functionT(target_user[:, 1:])
        negative_feature = self.functionT(negative_user[:, 1:])
        # 邻居部分嵌入高维
        user_feature_b = self.functionB(user[:, 1:])
        neighbor_feature_b = self.functionB(neighbor[:, 1:])

        # graph based feature
        user_graph_feature = self.emb(self.get_index(user))
        neighbor_graph_feature = self.emb(self.get_index(neighbor))
        target_graph_feature = self.emb(self.get_index(target_user))
        negative_graph_feature = self.emb(self.get_index(negative_user))

        # # Concat graph feature and user feature
        user_feature_concat = torch.cat((user_feature, user_graph_feature), dim=1)
        target_feature_concat = torch.cat((target_feature, target_graph_feature), dim=1)
        negative_feature_concat = torch.cat((negative_feature, negative_graph_feature), dim=1)
        #
        # # 计算target的差值
        distance_pos = self.diff(user_feature_concat, target_feature_concat)  # (batch, 1)
        #
        user_feature_neighbor = torch.cat((user_feature_b, user_graph_feature), dim=1)
        neighbor_feature_neighbor = torch.cat((neighbor_feature_b, neighbor_graph_feature), dim=1)
        #
        # # 训练模式
        distance_neg = self.diff(user_feature_concat, negative_feature_concat)  # 和负的差距
        distance_neighbor = self.diff(user_feature_neighbor, neighbor_feature_neighbor)  # 和邻居的差距
        # distance_pos = self.diff(user_graph_feature, target_graph_feature)  # (batch, 1)
        # distance_neg = self.diff(user_graph_feature, negative_graph_feature)  # 和负的差距
        # distance_neighbor = self.diff(user_graph_feature, neighbor_graph_feature)  # 和邻居的差距
        # if not self.train_mode:
        #     return self.output(distance_pos + distance_neighbor)
        output = self.output(distance_pos + distance_neighbor - distance_neg)
        return torch.sigmoid(output)
