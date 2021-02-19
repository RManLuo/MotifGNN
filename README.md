# MotifGNN
> Author Linhao Luo 
> luolinhao1998@gmail.com

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