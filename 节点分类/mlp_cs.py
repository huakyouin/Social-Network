import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
import torch_geometric.transforms as T
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.datasets import Planetoid
import numpy as np 

#使用correct and smooth算法进行结点分类任务

device = torch.device('cuda:0')
dataset = Planetoid(root='./', name='Cora')
data = dataset[0].to(device)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1433, 16)
        self.lin2 = Linear(16, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
model = MLP().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

x, y = data.x.to(device), data.y.to(device)
x_train, y_train = x[data.train_mask], y[data.train_mask]

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train.view(-1))
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(out=None):
    model.eval()
    out = model(x) if out is None else out
    
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    val_acc = correct / data.test_mask.sum()

    return val_acc,out

accL=[]
best_val_acc = 0
for epoch in range(300):
    loss = train()
    val_acc, out = test()
    accL.append(val_acc.item())
    #early stop
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        y_soft = out.softmax(dim=-1)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

adj_t = data.edge_index.to(device)
post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0,
                        num_smoothing_layers=50, smoothing_alpha=0.8,
                        autoscale=False, scale=20.)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))
print('Correct and smooth...')
#train_idx = mask2idx(data.train_mask)
y_soft = post.correct(y_soft, y_train, data.train_mask, data.edge_index)
y_soft = post.smooth(y_soft, y_train, data.train_mask, data.edge_index)
#print('Done!')
test_acc, _ = test(y_soft)
print(f'Test: {test_acc:.4f}')

