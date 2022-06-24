## Node Classification

### §1 环境部署
```bash
conda create --name sn python=3.8
conda activate sn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install dgl ogb matplotlib class-resolver
```


### §2 经典模型
- MLP
```bash
python mlp.py
```
best acc on test: 0.599 ,Epoch: 441



- Label Propagation
```bash
python lp.py
```
best acc on test: 0.7160000205039978 ,lp迭代次数: 12

- node2vec
```bash
python node2vec.py
```
best acc in test: 0.738 ,Epoch: 488

- GCNConv
```bash
python gcn.py
```
best acc on test: 0.8230000138282776 ,Epoch: 527

- Chebconv
```bash
python cheb.py
```
best acc on test: 0.8349999785423279 ,Epoch: 636

- GAT
```bash
python gat.py
```
best acc on test: 0.8180000185966492 ,Epoch: 69  结果不稳定

### §3 模型优化

- MLP+CS
```bash
python mlp_cs.py
```
best acc on test: 0.5490000247955322 ,Epoch: 55
Correct and smooth...
Test: 0.7770

- GCN+CS
```bash
python gcn_cs.py
```
best acc on test: 0.8199999928474426 ,Epoch: 19
Correct and smooth...
Best Test: 0.8270000219345093, correction_alpha: 0.5, smoothing_alpha: 0.8

- GCN add Mask
```bash
python -u gcn_addmask.py
```
best acc on test: 0.8340 ,Epoch: 594

- Teaching old MLPs
```bash
python glnn.py
```
Best Epoch:487, Train Acc:1.0000, Test Acc:0.8350

```bash
python glnn.py --teacher GCN2 --lamb 0.1
```
Best Epoch:340, Train Acc:1.0000, Test Acc:0.8550



