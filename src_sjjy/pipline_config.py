from neo4j import GraphDatabase

# neo4j connection
driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j"))
# random walk
k = 10  # Number of neighbors
pre_weight = 2  # Weight of return 
n = -1  # number of users to use, -1 means using all the users.
batch_size = 1000  # batchsize to save
cores = 12  # Multi threads
# Neo4j SQL to sample the motif.
motif_sql = '''
        match (a:User {{user_id: {id} }})<-[:msg|click]-(m)-[:msg|click]->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
        union
        match (a:User {{user_id: {id} }})-[:msg|click]->(m)<-[:msg|click]-(f)  return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n2}
        '''
"""
        match (a:User {{user_id: {id} }})<-[:msg|click]-(m)-[:msg|click]->(f) return "RESPOND" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n1}
        union
        match (a:User {{user_id: {id} }})<-->(m)-[:msg|click]->(f) return "SEND" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n1}
        union
        match (a:User {{user_id: {id} }})<-[:msg|click]-(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n2}
        union
        match (a:User {{user_id: {id} }})-[:msg|click]->(m)<-[:msg|click]-(f)  return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n2}
        union
        match (a:User {{user_id: {id} }})<-->(m)<-[:msg|click]-(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 2 as weight limit {n2}
        union
        match (a:User {{user_id: {id} }})-[:msg|click]->(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 3 as weight limit {n2}
        union
        match (a:User {{user_id: {id} }})<-->(m)<-->(f) return "DOUBLE" as r1, m.user_id as middle, f.user_id as final, 4 as weight limit {n2}
""" 
raw_walk_path = "../data/sjjy_data/motif_random_walk_path_M1+M4_b_{}.txt".format(pre_weight)  # Path of the raw random walk sequences
raw_emb_path = "../model/sjjy_motif_walk_M1+M4_b_{}.emb".format(pre_weight)  # Path of the raw embedding path
emb_save_path = "../model/sjjy_motif_walk_M1+M4_b_{}.emb".format(pre_weight)  # No need for data Sjjy
# motif random walk
raw_train_data_path = "../data/sjjy_data/train_data_v4.csv"  # train user pairs file path 原始的用户对id
raw_test_data_path = ""'../data/sjjy_data/test_data_v4.csv'  # test file path

train_data_path = "../data/sjjy_data/rec_data_train_M1+M4_b_{}.csv".format(pre_weight)  # train user pairs with neighbors
test_data_path = "../data/sjjy_data/rec_data_train_test_M1+M4_b_{}.csv".format(pre_weight)

# train
uid2idx_path = "../data/uid_2_idx.pkl" # user_id to id
model_save_path = "../model/recommend_M1+M4_b_{}.pb".format(pre_weight)  # final model save path
check_point_path = "../checkpoint/recommend_M1+M4_b_{}.pth".format(pre_weight)  # checkpoint path
feature_dict_path = "../data/sjjy_data/enc_feature_dict.pkl"
