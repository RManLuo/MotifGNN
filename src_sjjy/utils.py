#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 18:35
# @Author   : Raymond Luo
# @File     : utils.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import torch


class early_stop():
    def __init__(self, patient, min=True):
        self.min = min
        self.patient = patient
        self.current = 0
        self.bestloss = 1e5
        self.bestacc = 1e-5
        self.train = True

    def evulate(self, loss):
        if self.train:
            if self.min:
                if loss < self.bestloss:
                    self.bestloss = loss
                    self.current = 0

                else:
                    self.current += 1
                    if self.current > self.patient:
                        self.train = False
            else:
                if loss > self.bestacc:
                    self.bestacc = loss
                    self.current = 0
                else:
                    self.current += 1
                    if self.current > self.patient:
                        self.train = False

        return self.train


def binary_accuracy(y_true, y_pred):
    '''
    计算准确率
    :param y_true:
    :param y_pred:
    :return:
    '''
    t = torch.eq(y_true, torch.round(y_pred)).float()
    return torch.mean(t)

class build_dict():
    def __init__(self, userA, userB):
        self.all_user = []
        self.all_user.extend(userA)
        self.all_user.extend(userB)
        self.user_dict = {}
        for i in self.all_user:
            if i in self.user_dict:
                self.user_dict[i] += 1
            else:
                self.user_dict[i] = 1
        self.counts = len(self.user_dict)
        self.word_idx = dict(zip(self.user_dict.keys(), range(self.counts)))

    def transform(self, userlist):
        user_idx = []
        for i in userlist:
            if i in self.word_idx:
                user_idx.append(self.word_idx[i])
            else:
                raise Exception("No index in dict")
        return user_idx