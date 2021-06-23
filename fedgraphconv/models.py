import torch 
import torch.nn as nn 
from torch_geometric.nn import GCNConv, GATConv

class FFN(nn.Module): 
    def __init__(self, input_dim, num_class): 
        super().__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x): 
        x = self.dropout(torch.relu(self.layer1(x)))
        x = self.dropout(torch.relu(self.layer2(x)))
        x = self.dropout(torch.relu(self.layer3(x)))
        x = self.out(x)
        return x

class GCN_mhealth(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_mhealth, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier1 = nn.Linear(128, 64)
        self.classifier2 = nn.Linear(64, num_class)

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
        return out

class GCN_mhealth_Attn(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_mhealth_Attn, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(input_dim, 128, 4)
        self.fc1 = nn.Linear(128 * 4, 64)
        self.fc2 = nn.Linear(64, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        # Apply a final (linear) classifier.
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)

        return h

class GCN_wisdm(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_wisdm, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_class)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.dropout(h)
        # Apply a final (linear) classifier.
        h = self.bn1(h)
        h = self.fc2(h)
        h = h.relu()
        h = self.dropout(h)
        out = self.out(h)
        return out 

class GCN_wisdm_2conv(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_wisdm_2conv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 256)
        self.conv2 = GCNConv(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        # self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_class)
        

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.dropout(h)
        h = self.conv2(x, edge_index)
        h = h.relu()
        h = self.dropout(h)
        # Apply a final (linear) classifier.
        h = self.fc2(h)
        h = h.relu()
        h = self.dropout(h)
        out = self.out(h)
        return out 

class GCN_wisdm_3conv(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_wisdm_3conv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, 256)
        self.conv2 = GCNConv(input_dim, 256)
        self.conv3 = GCNConv(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        # self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_class)
        

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(x, edge_index)
        h = h.relu()
        h = self.conv3(x, edge_index)
        h = h.relu()
        # Apply a final (linear) classifier.
        h = self.fc2(h)
        h = h.relu()
        h = self.dropout(h)
        out = self.out(h)
        return out 

class GCN_wisdm_Attn(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(GCN_wisdm_Attn, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(input_dim, 1024, heads=1, dropout=0.5) #make sure to change later. 
        self.conv2 = GATConv(input_dim, 512, heads=1, dropout=0.5)
        self.conv3 = GATConv(input_dim, 256, heads=1, dropout=0.5)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256 * 1, 128)
        self.out = nn.Linear(128, num_class)    

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(x, edge_index)
        h = h.relu()
        h = self.conv3(x, edge_index)
        h = h.relu()
        # h = self.dropout(h)
        # Apply a final (linear) classifier.
        h = self.fc2(h)
        h = h.relu()
        # h = self.dropout(h)
        out = self.out(h)
        return out 


 