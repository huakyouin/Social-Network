import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from torch_geometric.nn import GCNConv

transform = T.Compose([
    # T.RandomNodeSplit('train_rest', num_val=500, num_test=500),
    T.TargetIndegree(),
])
dataset = Planetoid('./', 'Cora', transform=transform)
# dataset = Planetoid('./', 'Cora')
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()    
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)
    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    loss=F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

accL=[]
for epoch in range(600):
    loss=train()
    log = 'Epoch: {:03d}, loss:{:.4f}, Train acc: {:.4f}, Test cc: {:.4f}'
    t1,t2=test()
    if (epoch+1)%50==0:
        print(log.format(epoch+1,loss,t1,t2))
    accL.append(t2)

print('best acc on test: {:.4f}'.format(max(accL)),',Epoch:',accL.index(max(accL)))