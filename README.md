# MotifGNN
> Author Linhao Luo 
> luolinhao1998@gmail.com

Official code implementation for paper [A Motif-Based Graph Neural Network to Reciprocal Recommendation for Online Dating](https://link.springer.com/chapter/10.1007/978-3-030-63833-7_9)

## Environments
Ubuntu 20.04

python==3.6

GPU: RTX 2080

CUDA: 10.1

Database: Neo4j

Python pacakages

```
pytorch==1.6.0
pandas
neo4j
sklearn
numpy
gensim
tqdm
```

## Download the code

Due to the privacy issue, we only public the ssjy dataset. You can download the dataset [here](https://drive.google.com/file/d/1p_-piE7CUgLLr0xEsRk9i7RPhwKZ19hy/view?usp=sharing) and unzip to "data" folder.

## Data preprocessing

### Load graph to neo4j

```
sudo neo4j-admin import --database=sjjy.db --nodes:User=data/sjjy_data/users.csv --relationships=data/sjjy_data/relations.csv --id-type=INTEGER
```

### Generate train and test data

```
python3 data/prepare_data.py
```

### Preprocess data for training

```
python3 src_sjjy/data_preprocess.py
```

## Config

```
cat src_sjjy/pipline_config.py
```

## Train

```
python3 train_pipline.py
```

## Citation

```
@inproceedings{luo2020motif,
  title={A Motif-Based Graph Neural Network to Reciprocal Recommendation for Online Dating},
  author={Luo, Linhao and Liu, Kai and Peng, Dan and Ying, Yaolin and Zhang, Xiaofeng},
  booktitle={International Conference on Neural Information Processing},
  pages={102--114},
  year={2020},
  organization={Springer}
}
```
