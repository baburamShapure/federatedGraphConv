import torch 
import torch.nn as nn 
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(23, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier1 = nn.Linear(128, 64)
        self.classifier2 = nn.Linear(64, 12)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        h  = self.dropout(h)
        out = self.classifier1(h)
        out = self.dropout(out)
        out = self.classifier2(out)
        return out, h

model = GCN()

sum(p.numel() for p in model.parameters())
