import torch.nn.functional as F
from utils import *
from dataset import *
from layer import PEGConv, GATConv
from GAT_PE import GATConv_PE, GATConv_PE_v2
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, use_former_information = False):
        super(Net, self).__init__()
        
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_former_information = use_former_information
        
        self.conv1 = PEGConv(in_feats_dim = in_feats_dim, pos_dim = pos_dim, out_feats_dim = hidden_dim)
        self.conv2 = PEGConv(in_feats_dim = in_feats_dim, pos_dim = pos_dim, out_feats_dim = hidden_dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc = nn.Linear(2, 1)

    def forward(self, x, edge_index, idx):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class NetGAT(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, use_former_information = False):
        super(NetGAT, self).__init__()
        
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_former_information = use_former_information
        
        self.conv1 = GATConv(in_channels = in_feats_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.conv2 = GATConv(in_channels = in_feats_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc = nn.Linear(2, 1)

    def forward(self, x, edge_index, idx):

        pos_dim = self.pos_dim
        coors, feats = x[:, :pos_dim], x[:, pos_dim:]
        feats = self.conv1(feats, edge_index)
        feats = self.conv2(feats, edge_index)
        
        nodes_first = feats[idx[0]]
        nodes_second = feats[idx[1]]
        pos_first = coors[idx[0]]
        pos_second = coors[idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class NetGAT_PE(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, use_former_information = False):
        super(NetGAT_PE, self).__init__()
        
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_former_information = use_former_information
        
        self.conv1 = GATConv_PE(in_channels = in_feats_dim, pos_dim = pos_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.conv2 = GATConv_PE(in_channels = in_feats_dim, pos_dim = pos_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc = nn.Linear(2, 1)

    def forward(self, x, edge_index, idx):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

class NetGAT_PE_v2(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, use_former_information = False):
        super(NetGAT_PE_v2, self).__init__()
        
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_former_information = use_former_information
        
        self.conv1 = GATConv_PE_v2(in_channels = in_feats_dim, pos_dim = pos_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.conv2 = GATConv_PE_v2(in_channels = in_feats_dim, pos_dim = pos_dim, out_channels = hidden_dim, heads = 4, concat=False)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc = nn.Linear(2, 1)

    def forward(self, x, edge_index, idx):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
