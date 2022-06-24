import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv,SAGEConv
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU
device = torch.device('cuda:0')

class Net(torch.nn.Module):
    def __init__(self,dim=16):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='./', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01, weight_decay=5e-4)

def train_one_epoch():
    GCN.train()
    optimizer.zero_grad()
    out = GCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_one_epoch():
    GCN.eval()
    _, pred = GCN(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item()

accL=[]
GCN.train()
for epoch in range(601):
    loss = train_one_epoch()
    acc = test_one_epoch()
    accL.append(acc)
    if (epoch+1) % 100 == 0:
        print('epoch',epoch+1,'loss',loss,'test acc',acc)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#可视化
def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.title("GCN")
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
    plt.savefig('./GCN.png',dpi=600)
    
GCN.eval()
out = GCN(data)
visualize(out, color=data.y.detach().cpu().numpy())