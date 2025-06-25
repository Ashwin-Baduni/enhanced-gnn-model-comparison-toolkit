import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn import Linear

class ImprovedGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)
        self.classifier = Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index).relu()
        h1 = self.dropout(h1)
        h2 = self.conv2(h1, edge_index).relu()
        h2 = self.dropout(h2)
        h3 = self.conv3(h2, edge_index).relu()
        z = self.classifier(h3)
        return h3, z

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes=4, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=dropout)
        self.conv2 = GATConv(8*8, num_classes, heads=1, dropout=dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        z = self.conv2(h, edge_index)
        return h, z

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes=4):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 64)
        self.conv2 = SAGEConv(64, 32)
        self.classifier = Linear(32, num_classes)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index).relu()
        h2 = self.conv2(h1, edge_index).relu()
        z = self.classifier(h2)
        return h2, z
