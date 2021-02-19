#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/11/2 13:53
# @Author   : Raymond Luo
# @File     : eval.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import sqlite3
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import pickle
from utils import binary_accuracy
from mymodel_config import *
from tqdm import tqdm
import torch
from torch import nn
from gensim.models import KeyedVectors
# from recommend_model import recommend


def buid_all_features_dict(user_dict_orig):
    '''
    所有用户id的特征字典，index是user_id的整形
    :return:
    '''
    with open("../data/graph_data/joined_features_drop_dupli.txt") as f:
        features = f.readlines()

    def build_user_node(feature):
        l = feature.strip().split("\t")
        intlist = list(map(lambda x: int(x) if x != "NULL" else -1, l))  # 把null转换成-1
        # 将几个bigint搞掉
        intlist[10] = 0
        intlist[23] = 0
        intlist[24] = 0
        return intlist[0], intlist

    user_dict = {}
    for feature in features:

        user_id, l = build_user_node(feature)
        if user_id in user_dict_orig.word_idx:
            index = user_dict_orig.word_idx[user_id]
            l[0] = index
            user_dict[user_id] = l
    return user_dict


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv("../data/rec_data_train_test.csv")
db = sqlite3.connect('../data/feature_negative.db')
cursor = db.cursor()
with open(uid2idx_path, "rb") as f:
    uid2idx = pickle.load(f)
with open("../data/users.pkl", "rb") as f:  # user_id 2 index
    user_dict = pickle.load(f)  # index 是整形
features_dict = buid_all_features_dict(user_dict)
user_A = data["user"].values.tolist()[:eval_num]
user_B = data["target_user"].values.tolist()[:eval_num]
old_label = data['label'].map(lambda x: 1 if x == 0 else 0).values.tolist()[:eval_num]  # 旧模型0是推荐 1是不推荐
user_raw = []
target_raw = []
negative_raw = []
label_raw = []
for user, target, label in tqdm(zip(user_A, user_B, old_label), total=len(user_A)):
    if user not in features_dict or target not in features_dict:
        print("No user or target")
        continue
    user_features = features_dict[user]
    #  找他的负例用户
    find_sql = '''
                select negative_group from features where user_id='{st_user_id}'
                '''.format(st_user_id=user)
    cursor.execute(find_sql)
    result = cursor.fetchone()
    if result[0] != '[]':  # result 不为空
        clean_result = list(map(lambda x: int(x.strip()), result[0][1:-1].split(",")))  # 从数据库恢复并且变成int形式存储
    else:
        clean_result = []  # 空的时候跳过
        print("No negative")
        continue
    random_negative = random.choice(clean_result)
    if random_negative not in features_dict:
        print("No negative feeature")
        continue
    negative_features = features_dict[random_negative]  # 负例特征
    user_raw.append(user_features)  # 用户特征
    negative_raw.append(negative_features)
    target_feature = features_dict[target]  # 用户特征
    target_raw.append(target_feature)
    label_raw.append(label)
print(len(user_raw))
user, neighbor, target, negative, label = torch.FloatTensor(user_raw), torch.FloatTensor(user_raw), torch.FloatTensor(
    target_raw), torch.FloatTensor(negative_raw), torch.FloatTensor(label_raw)
dataset = Data.TensorDataset(user, neighbor, target, negative, label)
dataloader = Data.DataLoader(dataset, batch_size=1024, shuffle=False)

model = torch.load(model_save_path)
model.to(device)
model.eval()
running_loss = 0.0
running_acc = 0.0
val_size = len(dataloader)
criterion_binary = nn.BCELoss()
for i, data in enumerate(dataloader):
    user, neighbor, target, negative, label = data
    user = user.to(device)
    neighbor = neighbor.to(device)
    target = target.to(device)
    negative = negative.to(device)
    label = label.to(device)
    # Forward pass
    output = model(user, neighbor, target, negative)  # (batchsize, 1)

    # compute loss
    loss = criterion_binary(output.view(-1, 1), label.view(-1, 1))
    running_loss += loss.item()
    running_acc += binary_accuracy(label, output.view(-1, 1)).item() * 100
torch.cuda.empty_cache()  # 释放显存
print('ValLoss: {:.4f}, Acc_: {:.4f}'.format(running_loss / val_size, running_acc / val_size))
