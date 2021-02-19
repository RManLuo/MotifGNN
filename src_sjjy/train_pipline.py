import generate_walk_path
import motif_random_walk
import train
from neo4j import GraphDatabase
import pipline_config

if __name__ == "__main__":
    generate_walk_path.main(pipline_config)
    generate_walk_path.train_node2vec(pipline_config.raw_walk_path)
    # # generate_walk_path.change_emb_index(
    # #     pipline_config.raw_emb_path, pipline_config.uid2idx_path)
    motif_random_walk.main(pipline_config.raw_train_data_path,
                           pipline_config.train_data_path)
    motif_random_walk.main(pipline_config.raw_test_data_path,
                           pipline_config.test_data_path)
    train.train()
    train.eval()
