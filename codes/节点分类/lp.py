from torch_geometric.nn import LabelPropagation
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

device = torch.device('cuda:0')

dataset = Planetoid(root='./', name='Cora')
data = dataset[0].to(device)

accL=[]
group=range(3,18)
for num_layers in group:
    model = LabelPropagation(num_layers=num_layers, alpha=0.9) #num_layers控制lp迭代次数
    out = model(data.y, data.edge_index, mask=data.train_mask)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    accL.append(accuracy.item())
    print('test acc: ',accuracy.item(),'lp迭代次数:',num_layers)

print('best acc on test:',max(accL),',lp迭代次数:',accL.index(max(accL))+group[0])
