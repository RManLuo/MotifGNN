#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 11:29
# @Author   : Raymond Luo
# @File     : train_config.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import pipline_config

# config
batch_size = 1024
total_eopchs = 100
voc_dim = 128
total_node = 548394
continue_tranning = False
Kfold = 1  # 10  # k-fold
start_fold = 0  # 初始fold
model_save_path = pipline_config.model_save_path  # 最终模型保存路径
check_point_path = pipline_config.check_point_path  # checkpoint 路径
train_data_path = pipline_config.train_data_path
test_data_path = pipline_config.test_data_path
motif_save_path = pipline_config.emb_save_path
feature_dict_path = pipline_config.feature_dict_path
uid2idx_path = pipline_config.uid2idx_path
train_data_rate = 0.8
save_every = 1  # 每多少epoch 保存
patience = 2
neighbors_num = 10
result_save_path="./MotifGNN(M1, M4).txt"