# ReadMe
社交网络课程期末项目，包含：
- 基础任务，社区检测和网络分析
- 机器学习任务，节点分类
- 机器学习任务，链接预测


### §1 环境部署
```bash
conda create --name sn python=3.8 notebook
conda activate sn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 
conda install python-louvain 
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install dgl ogb matplotlib class-resolver infomap gensim tensorflow fastdtw
```