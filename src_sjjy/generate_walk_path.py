#!/usr/bin/python3
# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2019/11/2 13:23
# @Author   : Raymond Luo
# @File     : generate_gcn_walk_path.py
# @User     : luoli
# @Software: PyCharm
# Reference:**********************************************
import csv

import pickle
from neo4j import GraphDatabase
import argparse
from multiprocessing.dummy import Pool
import numpy as np
from tqdm import tqdm
from train_config import *
import csv
from neo4j import GraphDatabase
import argparse
from multiprocessing.dummy import Pool
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors, Word2Vec
import pipline_config
driver = pipline_config.driver
# driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j"))
# parser = argparse.ArgumentParser(description="Run motif random walk.")
# parser.add_argument("--k", type=int, default=10, help="Find k neighbors with random walk")
# parser.add_argument("--path", type=str, default="../data/motif_random_walk_path_M4_M6.txt", help="Path to save train data")
# parser.add_argument("--embpath", type=str,
#                     default="../model/motif_walk_M4_M6.emb")
# parser.add_argument("--emb_save_path", type=str,
#                     default="../model/motif_walk_idx_M4_M6.emb")
# parser.add_argument("--batch_size", type=int, default=10000, help="Batch size to save file")
# parser.add_argument("--cores", type=int, default=12, help="CPU cores")
# parser.add_argument("--pre_weight", type=int, default=2, help="Move back weight")
# parser.add_argument("--n", type=int, default=-1, help="Number of relation to select")
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
    motif_sql = pipline_config.motif_sql.format(id=cur, n1=pipline_config.k // 2, n2=pipline_config.k)  # M4_M6
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
        normalized_probs = [float(u_prob) / norm_const for u_prob in prob_list]  # 权重转换成概率
    if cur_neighbor:
        next_user = np.random.choice(cur_neighbor, p=normalized_probs)  # 选择下一个user
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


def build_train_set(user):
    walk = motif_random_walk(user, pipline_config.k)
    return walk


def save(f, list):
    for i in list:
        final = " ".join(str(a) for a in i)
        f.write(final)
        f.write("\n")


def main(args):
    if pipline_config.n > 0:
        user_sql = '''
            match (a:User) return a.user_id AS user limit {}
            '''.format(pipline_config.n)
    else:
        user_sql = '''
            match (a:User) return a.user_id AS user
            '''
    with driver.session() as session:
        result = session.run(user_sql)
        users = []  # 用户列表
        for v in result.values():
            users.append(v[0])
    cores = pipline_config.cores
    print("CPU cores: ", cores)
    tpool = Pool(cores)
    with open(pipline_config.raw_walk_path, 'w') as f:
        print("Building users walk....")
        train_set = []
        for res in tqdm(tpool.imap_unordered(build_train_set, users), total=len(users)):
            if res:
                train_set.append(res)
            if len(train_set) > pipline_config.batch_size:
                # 内容
                save(f, train_set)
                train_set = []
            # 最后还要再存一次
        if train_set:
            save(f, train_set)
    return


def train_node2vec(path):
    with open(path, "r") as f:
        walkfile = f.readlines()
    walk = []
    for line in walkfile:
        walk.append(line.strip("\n"))
    sentences = [s.split() for s in walk]
    model = Word2Vec(sentences, size=voc_dim, window=3, min_count=0, sg=1, workers=12, iter=200, compute_loss=True)
    print("Node2vec loss:", model.get_latest_training_loss())
    model.wv.save_word2vec_format(pipline_config.raw_emb_path)


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
        idx = uid2idx[int(line_list[0])]  # uid 2 idxr
        line_list[0] = str(idx)  # 转回去
        new_line = " ".join(line_list)
        new_file.append(new_line)
    with open(pipline_config.emb_save_path, "w", encoding="utf-8") as f:
        for line in new_file:
            f.write(line)



if __name__ == "__main__":
    main(pipline_config)
    train_node2vec(pipline_config.raw_walk_path)
    change_emb_index(pipline_config.raw_emb_path, "../data/uid_2_idx.pkl")
    driver.close()