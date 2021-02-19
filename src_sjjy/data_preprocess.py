#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/25 9:47
# @Author   : Raymond Luo
# @File     : data_preprocess.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
from neo4j import GraphDatabase
import pickle
import sqlite3
from tqdm import tqdm
import pandas as pd


def split(data_path, frac=0.8):
    data_all = pd.read_csv(data_path)
    data_all = data_all.sample(frac=1.)
    train_size = int(frac * len(data_all))
    train_data = data_all[:train_size]
    test_data = data_all[train_size:]
    print(len(train_data))  # 2610302
    print(len(test_data))  # 652576
    train_data.to_csv("../data/rec_data_train.csv", index=None)
    test_data.to_csv("../data/rec_data_train_test.csv", index=None)


def build_userdict():
    '''
    生成uid2idx和idx2uid的字典
    :return:
    '''
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j"))
    with driver.session() as session:
        all_node = session.run("match (a:User) return collect(a.user_id)")
        all_user = []
        user_dict = {}
        for i in all_node.values():
            all_user.extend(i[0])  # i[0] 是一个list
    for i in all_user:
        if i in user_dict:
            user_dict[i] += 1
        else:
            user_dict[i] = 1
    counts = len(user_dict)
    uid_2_idx = dict(zip(user_dict.keys(), range(counts)))
    idx_2_uid = {v: k for k, v in user_dict.items()}
    print("Total user:{}".format(len(uid_2_idx)))
    with open("../data/uid_2_idx.pkl", "wb") as f:  # key: int(uid), values: int(idx)
        pickle.dump(uid_2_idx, f)
    with open("../data/idx_2_uid.pkl", "wb") as f:
        pickle.dump(idx_2_uid, f)


def build_feature_dict():
    '''
    创建feature.db
    :return:
    '''
    db = sqlite3.connect('../data/features.db')
    cursor = db.cursor()
    create_table_sql = '''
        create table if not exists features
        (id integer PRIMARY KEY AUTOINCREMENT,
        user_id INT,
        features TEXT)
        '''
    cursor.execute(create_table_sql)  # 创建数据表
    with open("../data/uid_2_idx.pkl", "rb") as f:
        uid2idx = pickle.load(f)
    with open("../data/graph_data/joined_features_drop_dupli.txt") as f:
        features = f.readlines()
    insert_sql = '''
        insert into features
        (id, user_id, features)
        values (null, :st_user_id,:st_feature)
        '''
    for feature in tqdm(features):
        feature_list = feature.strip().split("\t")
        feature_list_int = list(map(lambda x: int(x) if x != "NULL" else -1, feature_list))  # 把null转换成-1
        user_id = int(feature_list_int[0])  # 取user_id 转int
        if user_id in uid2idx:
            cursor.execute(insert_sql, {'st_user_id': user_id, 'st_feature': str(feature_list_int)})
            # feature_dict[user_id] = {'feature': feature_list_int}
    db.commit()
    cursor.close()
    return


def feature_2_dict():
    with open("../data/uid_2_idx.pkl", "rb") as f:
        uid2idx = pickle.load(f)
    with open("../data/graph_data/joined_features_drop_dupli.txt") as f:
        features = f.readlines()
    feature_dict = {}
    for feature in tqdm(features):
        feature_list = feature.strip().split("\t")
        feature_list_int = list(map(lambda x: int(x) if x != "NULL" else -1, feature_list))  # 把null转换成-1， 并以int形式存储
        user_id = int(feature_list_int[0])  # 取user_id 转int
        if user_id in uid2idx:
            feature_dict[user_id] = feature_list_int
    print("Total user: {}".format(len(feature_dict)))
    with open("../data/feature_dict.pkl", "wb") as f:
        pickle.dump(feature_dict, f)


if __name__ == "__main__":
    # build_userdict()
    # build_feature_dict()
    # feature_2_dict()
    split("../data/train_data.csv")
