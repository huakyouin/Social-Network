import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

device = torch.device('cuda:0')


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1433, 32)
        self.lin2 = Linear(32, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin2(x)
        return x


dataset = Planetoid(root='./', name='Cora')
model = MLP().to(device)
data = dataset[0].to(device)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)  # Define optimizer.

def train_one_epoch():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x)  # Perform a single forward pass.
      # Compute the loss solely based on the training nodes.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward()   # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss.item()

def test_one_epoch():
      model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

accL=[]
model.train()
for epoch in range(1000):
    loss = train_one_epoch()
    acc = test_one_epoch()
    accL.append(acc)
    if (1+epoch) % 100 == 0:
        print('epoch',epoch+1,'loss',loss,'accuracy',acc)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))