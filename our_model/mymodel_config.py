#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 11:29
# @Author   : Raymond Luo
# @File     : train_config.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************

# config
eval_num=100
batch_size = 102400 // 4
total_eopchs = 50
voc_dim = 512
total_node = 228470
continue_tranning = False
Kfold = 1  # 10  # k-fold
start_fold = 0  # 初始fold
model_save_path = '../model/recommend_0_acc87.pb'  # 最终模型保存路径
check_point_path = "../checkpoint/gcn.pth"  # checkpoint 路径
train_data_path = "../data/train_data_gcn.csv"
test_data_path = "../data/rec_data_train_test.csv"
feature_dict_path = "../data/feature_dict.pkl"
uid2idx_path = "../data/uid_2_idx.pkl"
train_data_rate = 0.8
save_every = 1  # 每多少epoch 保存
patience = 2
neighbors_num = 3
