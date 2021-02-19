#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/10/24 9:03
# @Author   : Raymond Luo
# @File     : motif_random_walk.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
# 执行motif版本的random walk，返回 (user, Na, target, Nt)的形式
import csv
from neo4j import GraphDatabase
import argparse
from multiprocessing.dummy import Pool
import numpy as np
from tqdm import tqdm
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
import pickle
import pipline_config
driver = pipline_config.driver
# driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j"))
# parser = argparse.ArgumentParser(description="Run motif random walk.")
# parser.add_argument("--k", type=int, default=10,
#                     help="Find k neighbors with random walk")
# parser.add_argument("--path", type=str,
#                     default="../data/train_data_M1_7.csv", help="Path to save train data")
# parser.add_argument("--embpath", type=str,
#                     default="../model/motif_walk_M1_7.emb")
# parser.add_argument("--emb_save_path", type=str,
#                     default="../model/motif_walk_idx_M1_7.emb")
# parser.add_argument("--batch_size", type=int, default=10000,
#                     help="Batch size to save file")
# parser.add_argument("--cores", type=int, default=12, help="CPU cores")
# parser.add_argument("--pre_weight", type=int,
#                     default=2, help="Move back weight")
# parser.add_argument("--n", type=int, default=10000,
#                     help="Number of relation to select")
# args = parser.parse_args()


def walk(pre, cur):
    '''
    找下一个walk的节点
    :param pre: 之前的节点
    :param cur: 当前的节点
    :return: next_user : 下一个节点
    '''
    # motif_sql = '''
    #     match (a:User {{user_id:"{id}"}})<--(m)-->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
    #     union
    #     match (a:User {{user_id:"{id}"}})<-->(m)-->(f) return "SEND" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n1}
    #     union
    #     match (a:User {{user_id:"{id}"}})<--(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n2}
    #     union
    #     match (a:User {{user_id:"{id}"}})-->(m)<--(f)  return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n2}
    #     union
    #     match (a:User {{user_id:"{id}"}})<-->(m)<--(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n2}
    #     union
    #     match (a:User {{user_id:"{id}"}})-->(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n2}
    #     union
    #     match (a:User {{user_id:"{id}"}})<-->(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 4 as weight limit {n2}
    #     '''.format(id=cur, n1=args.k // 2, n2=args.k)  # M1-M7
    # motif_sql = '''
    #     match (a:User {{user_id:"{id}"}})<-[r1:SEND]-(m)-[r2:SEND]->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
    #     union
    #     match (a:User {{user_id:"{id}"}})-[r1:SEND]->(m)<-[r2:SEND]-(f) return "SEND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
    #     '''.format(id=cur, n1=args.k // 2, n2=args.k)  # M1+M4
    # motif_sql = '''
    #     match (a:User {{user_id:"{id}"}})<-[r1:SEND]->(m)<-[r2:SEND]->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 4 as weight limit {n1}
    #     '''.format(id=cur, n1=args.k // 2, n2=args.k)  # M7
    # motif_sql = '''
    #     match (a:User {{user_id:"{id}"}})<--(m)-->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
    #     union
    #     match (a:User {{user_id:"{id}"}})<-->(m)-->(f) return "SEND" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n1}
    #     union
    #     match (a:User {{user_id:"{id}"}})<--(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n2}'''.format(id=cur, n1=args.k // 2, n2=args.k)  # M1_M3
    motif_sql = pipline_config.motif_sql.format(id=cur, n1=pipline_config.k // 2, n2=pipline_config.k)
    if pre:  # 不是起始节点
        cur_neighbor = [pre]  # 当前的邻居节点list, 添加之前的节点
        prob_list = [pipline_config.pre_weight]  # 转移概率表
    else:
        cur_neighbor = []  # 当前的邻居节点list, 添加之前的节点
        prob_list = []  # 转移概率表
    with driver.session() as session:
        result = session.run(motif_sql)  # 找到5种motif
        for v in result.values():  # (r1, middle, final, weight)
            cur_neighbor.append(v[2])
            prob_list.append(v[3])
        norm_const = sum(prob_list)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in prob_list]  # 权重转换成概率
    if cur_neighbor:
        next_user = np.random.choice(
            cur_neighbor, p=normalized_probs)  # 选择下一个user
        return next_user
    else:
        return cur  # 没有邻居


def motif_random_walk(src, k):
    '''
    motif random walk
    :param pre: 之前的节点
    :param src: 开始节点的userid
    :param k: 找多少个
    :return: walk list, len=k
    '''
    walk_list = [src]
    while len(walk_list) < k:
        cur = walk_list[-1]
        if len(walk_list) == 1:
            next = walk(None, cur)
        else:
            prev = walk_list[-2]
            next = walk(prev, cur)
        walk_list.append(next)
    return walk_list


def build_train_set(pair_list):
    result = []
    userA, userB, ground_truth = pair_list
    walk_A = motif_random_walk(userA, pipline_config.k)
    walk_B = motif_random_walk(userB, pipline_config.k)
    label = ground_truth
    result.append([userA, userB, walk_A, walk_B, label])
    # for userA, userB in pair_list:  # 遍历所有双向关系
    #     walk_A = motif_random_walk(userA, args.k)
    #     walk_B = motif_random_walk(userB, args.k)
    #     label = ground_truth
    #     result.append([userA, userB, walk_A, walk_B, label])
    return result


def main(raw_file_path, save_file_path):
    # pos_sql = '''
    #         match (a:User)-[:SEND]->(b:User) where (b)-[:RESPOND]->(a) return a.user_id as A, b.user_id as B
    #         '''
    # with driver.session() as session:
    #     result = session.run(pos_sql)  # 找到所有的双向关系
    #     pos_pair_list = []
    #     for v in result.values():
    #         pos_pair_list.append(v + [1])  # [A, B, 1]
    # neg_sql = "match (a)-[:SEND]->(b) where not (b)-[:RESPOND]->(a) return a.user_id, b.user_id limit {}".format(
    #     len(pos_pair_list))  # 单向关系
    # with driver.session() as session:
    #     result = session.run(neg_sql)  # 找到所有的单向关系
    #     neg_pair_list = []
    #     for v in result.values():
    #         neg_pair_list.append(v + [0])  # [A, B, 0]
    cores = pipline_config.cores
    data=pd.read_csv(raw_file_path)
    pair_list=data.values.tolist()
    print("CPU cores: ", cores)
    tpool = Pool(cores)
    with open(save_file_path, "w", encoding='utf8', newline='') as outFileCsv:
        writer = csv.writer(outFileCsv)
        # 表头
        head = ['user', 'target_user', 'user_neighbor',
                'target_neighbor', 'label']
        writer.writerow(head)
        print("Building datasaet....")
        train_set = []
        for res in tqdm(tpool.imap_unordered(build_train_set, pair_list), total=len(pair_list)):
            if res:
                train_set.extend(res)
            if len(train_set) > pipline_config.batch_size:
                # 内容
                writer.writerows(train_set)
                train_set = []
        # 最后还要再存一次
        if train_set:
            writer.writerows(train_set)
    return


def train_motif_wordemb(path):
    data = pd.read_csv(path)
    walk_a = data['user_neighbor'].values.tolist()
    walk_b = data['target_neighbor'].values.tolist()
    walk_a.extend(walk_b)
    walk = []
    for line in walk_a:
        new_line = line[1:-1].split(", ")
        walk.append(new_line)
    model = Word2Vec(walk, size=128, window=3, min_count=0,
                     sg=1, workers=12, iter=2, compute_loss=True)
    print("Node2vec loss:", model.get_latest_training_loss())
    model.wv.save_word2vec_format(args.embpath)


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
    with open(args.emb_save_path, "w", encoding="utf-8") as f:
        for line in new_file:
            f.write(line)


def split(data_path, frac=0.8):
    data_all = pd.read_csv(data_path)
    data_all = data_all.sample(frac=1.)
    train_size = int(frac * len(data_all))
    train_data = data_all[:train_size]
    test_data = data_all[train_size:]
    print(len(train_data))  # 2610302
    print(len(test_data))  # 652576
    train_data.to_csv("../data/rec_data_train_M1_4.csv", index=None)
    test_data.to_csv("../data/rec_data_train_test_M1_4.csv", index=None)


if __name__ == "__main__":
    main("../data/train_data.csv","../data/rec_data_train_M4_M6.csv")
    main('../data/test_data.csv',"../data/rec_data_train_test_M4_M6.csv")
    driver.close()  # 关闭连接
    # train_motif_wordemb(args.path)
    # change_emb_index(args.embpath, "../data/uid_2_idx.pkl")
    # split(args.path)
