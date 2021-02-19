#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 10:55
# @Author   : Raymond Luo
# @File     : train.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import torch
import pickle
import torch.utils.data as Data
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from train_config import *
import pandas as pd
from model import rec_model
from utils import early_stop, binary_accuracy
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


class rec_dataset(Data.Dataset):
    def __init__(self, train_data, uid2idx, feature_dict):
        self.data = pd.read_csv(train_data).values
        with open(uid2idx, "rb") as f:
            self.uid2idx = pickle.load(f)
        with open(feature_dict, "rb") as f:
            self.feature_dict = pickle.load(f)

    def __getitem__(self, item):
        data = self.data[item]
        userA, userB, neighborA, neighborB, label = data
        userA_feature = self.feature_dict[userA]
        userB_feature = self.feature_dict[userB]
        # userA_feature[0] = self.uid2idx[userA]  # uid2idx
        # userB_feature[0] = self.uid2idx[userB]  # uid2idx

        neighborA_feature = self.build_neighbor(neighborA)
        neighborB_feature = self.build_neighbor(neighborB)
        return torch.FloatTensor(userA_feature), \
               torch.FloatTensor(userB_feature), \
               torch.FloatTensor(neighborA_feature), \
               torch.FloatTensor(neighborB_feature), \
               torch.FloatTensor([label])

    def mask(self, feature):
        '''
        搞掉几个bigint
        :param feature:
        :return:
        '''
        # 将几个bigint搞掉
        feature[10] = 0
        feature[23] = 0
        feature[24] = 0
        return feature

    def build_neighbor(self, neigbhor):
        '''
        huanyuan
        :param neigbor: neighbor str list
        :return: neighgbor * feature
        '''
        neigbhor = neigbhor[1:-1].split(", ")  # 去掉[]，并转换成list
        feature_list = []
        for user in neigbhor:
            user = int(user)  # str 2 int
            user_feature = self.feature_dict[user]
            # user_feature[0] = self.uid2idx[user]
            feature_list.append(user_feature)
        return feature_list

    def __len__(self):
        return len(self.data)


def val(model, dataloader, criterion_binary, ear, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    predict = []
    ground_truth = []
    val_size = len(dataloader)
    for i, data in enumerate(dataloader):
        user, target, neighborA, neighborB, label = data
        with torch.no_grad():
            user = user.to(device)
            target = target.to(device)
            neighborA = neighborA.to(device)
            neighborB = neighborB.to(device)
            label = label.to(device)

            # Forward pass
            output = model(user, target, neighborA, neighborB)
            output = torch.sigmoid(output)
            predict.append(output.cpu().detach().numpy())
            ground_truth.append(label.cpu().detach().numpy())
        # compute loss
        loss = criterion_binary(output.view(-1, 1), label)
        running_loss += loss.item()
        running_acc += binary_accuracy(label, output.view(-1, 1)).item() * 100
    torch.cuda.empty_cache()  # 释放显存
    print('ValLoss: {:.4f}, Acc_: {:.4f}'.format(running_loss / val_size, running_acc / val_size))
    writer.add_scalar('Accuracy/Val', running_acc / val_size, epoch)
    writer.add_scalar('Loss/Val', running_loss / val_size, epoch)
    predict = np.concatenate(predict, axis=0).flatten()
    ground_truth = np.concatenate(ground_truth, axis=0).flatten()
    np.savetxt(result_save_path, (predict, ground_truth))
    print(classification_report(ground_truth,
                                np.where(predict > 0.5, 1, 0), digits=4))
    auc = roc_auc_score(ground_truth, predict)
    print(auc)
    model.train()
    train_status = ear.evulate(auc)
    return train_status


def train():
    # config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()  # log每两分钟保存到runs文件夹

    # dataset
    train_dataset = rec_dataset(train_data_path, uid2idx_path, feature_dict_path)
    test_dataset = rec_dataset(test_data_path, uid2idx_path, feature_dict_path)
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Building dataloader finished")
    print("Total train length: {} test length: {}".format(len(train_dataset), len(test_dataset)))

    # model and evaluation
    model = rec_model(node_size=total_node, node_dim=voc_dim, neighbors_num=neighbors_num)
    criterion_binary = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    global_step = 0
    if continue_tranning:
        checkpoint = torch.load(check_point_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['global_step']

    # train
    print_every = len(train_dataloader) // 10  # 一个epoch打印几次
    ear = early_stop(patience, min=False)
    for epoch in range(total_eopchs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_dataloader):
            user, target, neighborA, neighborB, label = data
            user = user.to(device)
            target = target.to(device)
            neighborA = neighborA.to(device)
            neighborB = neighborB.to(device)
            label = label.to(device)
            # Forward pass
            optimizer.zero_grad()
            output = model(user, target, neighborA, neighborB)
            output = torch.sigmoid(output)
            # Compute loss
            loss = criterion_binary(output.view(-1, 1), label)
            loss.backward()
            optimizer.step()
            acc = binary_accuracy(label, output.view(-1, 1)).item() * 100
            running_loss += loss.item()
            running_acc += acc
            if i % print_every == 0:
                print("Step: {} Loss: {} ACC: {} ".format(i, loss.item(), acc))

        print("Epoch:{}/{} Loss: {} ACC: {}".format(epoch, total_eopchs, running_loss / len(train_dataloader),
                                                    running_acc / len(train_dataloader)))
        writer.add_scalar('Accuracy/train', running_acc / len(train_dataloader), global_step)
        writer.add_scalar('Loss/train', running_loss / len(train_dataloader), global_step)
        if epoch % save_every == 0:
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step
            }
            torch.save(state, check_point_path)
        train_status = val(model, test_dataloader, criterion_binary, ear, device, writer, global_step)
        global_step += 1  # 全局步数++
        if not train_status:
            break
    torch.save(model, model_save_path)  # 最终保存


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
        with torch.no_grad():
            # Forward pass
            score = model(user, target, neighborA, neighborB)
            score = torch.sigmoid(score)
        predict.append(score.cpu().detach().numpy())
        ground_truth.append(label.cpu().detach().numpy())
    predict = np.concatenate(predict, axis=0).flatten()
    ground_truth = np.concatenate(ground_truth, axis=0).flatten()
    np.savetxt("mymodel_result.txt", (predict, ground_truth))
    print(classification_report(ground_truth,
                                np.where(predict > 0.5, 1, 0), digits=5))
    print(roc_auc_score(ground_truth, predict))


if __name__ == "__main__":
    train()
    eval()
