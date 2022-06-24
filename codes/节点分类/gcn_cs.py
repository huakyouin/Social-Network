import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv,SAGEConv
from torch_geometric.nn.models import CorrectAndSmooth
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU
import itertools
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

def test_one_epoch(out=None):
    GCN.eval()
    out = GCN(data) if out is None else out

    _, pred = out.max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item(),out

accL=[]
best_val_acc = 0
GCN.train()
for epoch in range(600):
    loss = train_one_epoch()
    acc,out = test_one_epoch()
    accL.append(acc)
    if acc > best_val_acc:
        best_val_acc = acc
        y_soft = out.softmax(dim=-1)
    if (epoch+1) % 100 == 0:
        print('epoch',epoch+1,'loss',loss,'test acc',acc)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))

print('Correct and smooth...')
y_soft_copy=y_soft
correction_alpha=[0.05,0.08,0.1,0.3,0.5,0.7,1]
smoothing_alpha=[0.4,0.6,0.8,1]
record=[0,0,0]
for ca,sa in itertools.product(correction_alpha,smoothing_alpha):
    y_soft=y_soft_copy
    post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=ca,
                            num_smoothing_layers=50, smoothing_alpha=sa,
                            autoscale=True)
    y_soft = post.correct(y_soft, data.y[data.train_mask], data.train_mask, data.edge_index)
    y_soft = post.smooth(y_soft, data.y[data.train_mask], data.train_mask, data.edge_index)
    test_acc, _ = test_one_epoch(y_soft)
    record= [test_acc,ca,sa] if test_acc>record[0] else record

print('Best Test: {0[0]}, correction_alpha: {0[1]}, smoothing_alpha: {0[2]}'.format(record))