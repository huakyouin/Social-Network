# Implementation of:
# Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation
# 对teacher进行优化，正则化优化

import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN, MLP,GCN2Conv,Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')

parser.add_argument('--teacher', type=str, default='GCN',help='choose the teacher model|GCN\GCN2')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = Planetoid('./', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.teacher=='GCN':
    gnn = GCN(dataset.num_node_features, hidden_channels=16,
          out_channels=dataset.num_classes, num_layers=2).to(device)
elif args.teacher=='GCN2':
    ## 再开一个teacher用的数据集
    transform = T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()])
    dataset2 = Planetoid(root='./', name='Cora',transform=transform)
    data2 = dataset2[0].to(device)
    ## GCN2 类定义
    class Net(torch.nn.Module):
        def __init__(self, hidden_channels, num_layers, alpha, theta,
                    shared_weights=True, dropout=0.0):
            super(Net, self).__init__()

            self.lins = torch.nn.ModuleList()
            self.lins.append(Linear(dataset.num_features, hidden_channels))
            self.lins.append(Linear(hidden_channels, dataset.num_classes))

            self.convs = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.convs.append(
                    GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                            shared_weights, normalize=False))

            self.dropout = dropout

        def forward(self, data):
            x, adj_t = data.x, data.adj_t
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_0 = self.lins[0](x).relu()

            for conv in self.convs:
                x = F.dropout(x, self.dropout, training=self.training)
                x = conv(x, x_0, adj_t)
                x = x.relu()

            x = F.dropout(x, self.dropout, training=self.training)
            x = self.lins[1](x)
            return x.log_softmax(dim=-1)
    gnn = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.4).to(device)
    data2.adj_t = gcn_norm(data2.adj_t)  # GCN标准化作为预处理手段
        
# mlp = MLP([dataset.num_node_features, 64, dataset.num_classes], dropout=0.6,
#           batch_norm=False).to(device)
mlp = MLP([dataset.num_node_features, 32, dataset.num_classes], dropout=0.6,
          batch_norm=True).to(device)

gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=1e-3)


def train_teacher():
    gnn.train()
    gnn_optimizer.zero_grad()
    if args.teacher=='GCN':
        out = gnn(data.x, data.edge_index)
    elif args.teacher=='GCN2':
        out =gnn(data2)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    gnn_optimizer.step()
    return float(loss)


@torch.no_grad()
def test_teacher():
    gnn.eval()
    if args.teacher=='GCN':
        pred = gnn(data.x, data.edge_index).argmax(dim=-1)
    elif args.teacher=='GCN2':
        pred = gnn(data2).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

record=[0,0,0]
print('Training Teacher GNN:')
for epoch in range(1, 201):
    loss = train_teacher()
    train_acc, val_acc, test_acc = test_teacher()
    record=[epoch,train_acc,test_acc]  if test_acc>record[2] else record
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
print('Best Epoch:{0[0]:03d}, Train Acc:{0[1]:.4f}, Test Acc:{0[2]:.4f}'.format(record))

with torch.no_grad():  # Obtain soft labels from the GNN:
    if args.teacher=='GCN':
        y_soft = gnn(data.x, data.edge_index).log_softmax(dim=-1)
    elif args.teacher=='GCN2':
        y_soft = gnn(data2).log_softmax(dim=-1)


def train_student():
    mlp.train()
    mlp_optimizer.zero_grad()
    out = mlp(data.x)
    loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss2 = F.kl_div(out.log_softmax(dim=-1), y_soft, reduction='batchmean',
                     log_target=True)
    loss = args.lamb * loss1 + (1 - args.lamb) * loss2
    loss.backward()
    mlp_optimizer.step()
    return float(loss)


@torch.no_grad()
def test_student():
    mlp.eval()
    pred = mlp(data.x).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

record=[0,0,0]
print('Training Student MLP:')
for epoch in range(1, 201):
    loss = train_student()
    train_acc, val_acc, test_acc = test_student()
    record=[epoch,train_acc,test_acc]  if test_acc>record[2] else record
    if epoch % 20 == 0:   
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print('Best Epoch:{0[0]:03d}, Train Acc:{0[1]:.4f}, Test Acc:{0[2]:.4f}'.format(record))



# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# #可视化
# def visualize(out, color):
#     z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
#     plt.figure(figsize=(10,10))
#     plt.xticks([])
#     plt.yticks([])

#     plt.title("GLNN(GCNII)")
#     plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     plt.show()
#     plt.savefig('./GLNN_GCNII.png',dpi=600)
    
# mlp.eval()
# out = mlp(data.x)
# visualize(out, color=data.y.detach().cpu().numpy())