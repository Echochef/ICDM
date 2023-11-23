# ICDM 2023 解决方案
**基于掩模图自编码器的预训练-微调方法**
**队伍：Echoch**

# 运行环境依赖

+ OS                            Ubuntu 22.04.1 LTS
+ CUDNN                         8.6.0.163
+ CUDA                          11.6
+ Python                        3.9.12
+ torch                         1.12.0+cu116
+ torch-cluster                 1.6.0
+ torch_geometric               2.4.0
+ torch-scatter                 2.0.9
+ torch-sparse                  0.6.15
+ torch-spline-conv             1.2.1
+ tqdm                          4.64.0
+ numpy                         1.21.5
+ pandas                        1.5.3
+ ogb                           1.3.6
+ scikit-learn                  1.0.2

# 代码复现流程
可直接`sh run.sh`进行一步复现，也可按照下述步骤一步步进行

注：即使是同样的随机种子，不同的机器Kmeans最终运行结果也差异较大，因此如果结果差异较大，可以在最后聚类的步骤多运行几次
```bash
python cluster.py --embedding_path embedding_arxiv.pt --output submit.txt --k 15 --seed -1 --runs 5
```
其中`--seed -1`为不指定随机种子

### (Step 1 & 2) 预训练 (arXiv数据集)

运行代码后会自动从ogb下载arxiv数据集
+ **(Step 1) 监督式预训练**
```bash
python pretrain-sup.py --dataset arxiv --save_path encoder_arxiv_sup.pt --root data
```
参数说明：
+ `dataset`: 预训练的数据集，可选arxiv和mag，本方案采用的是arxiv
+ `save_path`:预训练后模型参数的保存路径
+ `root`: 预训练数据集的存放目录，这里设置为当前目录下的`data/`文件夹

运行结束后目录下会出现`encoder_arxiv_sup.pt`模型参数文件

+ **(Step 2) 自监督式预训练**
```bash
python pretrain.py --dataset arxiv --save_path encoder_arxiv.pt --pretrain_path encoder_arxiv_sup.pt --lr 0.005 --root data
```
参数说明：
+ `dataset`: 预训练的数据集，可选arxiv和mag
+ `save_path`: 预训练后模型参数的保存路径
+ `pretrain_path`: 上一次（监督式）预训练参数保存的路径
+ `root`: 预训练数据集的存放目录，这里设置为当前目录下的`data/`文件夹

运行结束后目录下会出现`encoder_arxiv.pt`模型参数文件

### (Step 3) 加载预训练参数+微调 (ICDM数据集)

需下载数据集到`data/icdm2023_session1_test`文件夹内
```bash
python finetune.py --epochs 5 --pretrain_path encoder_arxiv.pt --embedding_save_path embedding_arxiv.pt --root data/icdm2023_session1_test
```
参数说明：
+ `epochs`: 微调轮次
+ `pretrain_path`: 上一次（自监督式）预训练参数保存的路径
+ `embedding_save_path`: 模型输出embedding的保存路径
+ `root`: ICDM比赛数据集的存放目录，这里设置为当前目录下的`data/icdm2023_session1_test`文件夹

运行结束后目录下会出现`embedding_arxiv.pt`的节点embedding文件

### (Step 4) 聚类+集成

```bash
python cluster.py --embedding_path embedding_arxiv.pt --output submit.txt --k 15 --seed 666 --runs 5
```
参数说明：
+ `embedding_save_path`: 模型输出embedding的保存路径
+ `output`: 模型输出聚类结果的保存路径，这里设置为当前目录下的`submit.txt`文件
+ `k`: 聚类类别数目
+ `seed`: 随机种子，这里指定的是`666`，也可以不指定随机种子或者指定为`-1`
+ `runs`: 聚类结果集成轮次

运行结束后目录下会出现`submit.txt`结果文件


# 文件夹目录

```
Code
├── fast_pytorch_kmeans
│   ├── init_methods.py
│   ├── __init__.py
│   ├── kmeans.py
│   ├── multi_kmeans.py
│   └── util.py
├── maskgae
│   ├── loss.py
│   ├── mask.py
│   ├── model.py
├── pretrain.py
├── pretrain-sup.py
├── finetune.py
├── cluster.py
├── README.md
├── run.sh
├── data
    └── icdm2023_session1_test
        ├── icdm2023_session1_test_edge.txt
        └── icdm2023_session1_test_node_feat.txt
    └── ogbn_arxiv
        ├── mapping
        │   ├── labelidx2arxivcategeory.csv.gz
        │   ├── nodeidx2paperid.csv.gz
        │   └── README.md
        ├── processed
        │   ├── data_processed
        │   ├── geometric_data_processed.pt
        │   ├── pre_filter.pt
        │   └── pre_transform.pt
        ├── raw
        │   ├── edge.csv.gz
        │   ├── node-feat.csv.gz
        │   ├── node-label.csv.gz
        │   ├── node_year.csv.gz
        │   ├── num-edge-list.csv.gz
        │   └── num-node-list.csv.gz
        ├── RELEASE_v1.txt
        └── split
            └── time
                ├── test.csv.gz
                ├── train.csv.gz
                └── valid.csv.gz
```
