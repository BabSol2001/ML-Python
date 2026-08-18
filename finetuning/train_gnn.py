import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

# ==========================================
# ۱. تعریف معماری مدل GraphSAGE
# ==========================================
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(LinkPredictor, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.decode(z, edge_label_index)

# ==========================================
# ۲. ساخت داده‌ها و فرآیند Fine-Tuning
# ==========================================
def create_sample_graph() -> Data:
    x = torch.randn((5, 16), dtype=torch.float)
    edge_index = torch.tensor([
        [0, 1, 0, 3, 4],
        [1, 2, 2, 0, 1]
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def fine_tune_model(data: Data, epochs: int = 50) -> LinkPredictor:
    # اطمینان از عدم None بودن داده‌ها جهت رفع خطای Type Checker
    assert data.edge_index is not None, "Data object must contain edge_index"
    assert data.x is not None, "Data object must contain node features (x)"

    model = LinkPredictor(in_channels=16, hidden_channels=32, out_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        )

        edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
        
        edge_label = torch.cat([
            torch.ones(data.edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model(data.x, data.edge_index, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        
        loss.backward()
        optimizer.step()

    model.eval()
    return model

if __name__ == "__main__":
    print("🚀 در حال ساخت گراف و شروع Fine-Tuning...")
    graph_data = create_sample_graph()
    tuned_model = fine_tune_model(graph_data)

    # ذخیره وزن‌های مدل و داده‌های گراف روی دیسک
    torch.save(tuned_model.state_dict(), "gnn_model.pth")
    torch.save(graph_data, "graph_data.pt")
    print("✅ مدل با موفقیت آموزش دید و در فایل‌های 'gnn_model.pth' و 'graph_data.pt' ذخیره شد.")