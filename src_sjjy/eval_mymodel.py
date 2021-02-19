#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/12/28 13:39
# @Author   : Raymond Luo
# @File     : eval_mymodel.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
from sklearn.metrics import classification_report, roc_auc_score
import os
import torch
import pickle
import torch.utils.data as Data
from tqdm import tqdm
from train_config import *
from train import rec_dataset
from model import rec_model
import numpy as np


def to_device(data_list, device):
    '''
    Feed data into device
    '''
    data2device = []
    for data in data_list:
        data2device.append(data.to(device, non_blocking=True))
    return data2device


def eval():
    # config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    # train_dataset = rec_dataset(train_data_path, uid2idx_path, feature_dict_path)
    test_dataset = rec_dataset(test_data_path, uid2idx_path, feature_dict_path)
    # train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = Data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Building dataloader finished")
    # print("Total train length: {} test length: {}".format(len(train_dataset), len(test_dataset)))

    # model and evaluation
    model = rec_model(node_size=total_node, node_dim=voc_dim,
                      neighbors_num=neighbors_num)
    model.to(device)
    global_step = 0
    checkpoint = torch.load(check_point_path)
    model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    global_step = checkpoint['global_step']

    model.eval()
    predict = []
    ground_truth = []
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        user, target, neighborA, neighborB, label = data
        user = user.to(device)
        target = target.to(device)
        neighborA = neighborA.to(device)
        neighborB = neighborB.to(device)
        label = label.to(device)

        # Forward pass
        score = model(user, target, neighborA, neighborB)
        predict.append(score.cpu().detach().numpy())
        ground_truth.append(label.cpu().detach().numpy())
    predict = np.concatenate(predict, axis=0).flatten()
    ground_truth = np.concatenate(ground_truth, axis=0).flatten()
    np.savetxt("mymodel_result.txt", (predict, ground_truth))
    print(classification_report(ground_truth,
                                np.where(predict > 0.5, 1, 0), digits=5))
    print(roc_auc_score(ground_truth, predict))


eval()
