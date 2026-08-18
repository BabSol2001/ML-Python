import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

# ---------------------------------------------------------
# ۱. ساختار مدل PyG
# ---------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ToolPredictorPyG(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# ---------------------------------------------------------
# ۲. ساخت داده و آموزش (Fine-Tuning)
# ---------------------------------------------------------
def train_and_save():
    x = torch.randn((5, 8), dtype=torch.float)
    edge_index = torch.tensor([[0, 0, 1, 2], [3, 4, 3, 4]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    # 🔹 حل خطای Type Checker: اطمینان دادن از عدم None بودن داده‌ها
    assert data.edge_index is not None, "Data object must contain edge_index"
    assert data.x is not None, "Data object must contain node features (x)"

    model = ToolPredictorPyG(in_channels=8, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        
        neg_edge = negative_sampling(data.edge_index, num_nodes=5, num_neg_samples=4)
        edge_label_index = torch.cat([data.edge_index, neg_edge], dim=-1)
        edge_label = torch.cat([torch.ones(4), torch.zeros(4)], dim=0)

        z = model.encoder(data.x, data.edge_index)
        out = model.decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "pyg_tool_model.pth")
    torch.save(data, "pyg_data.pt")
    print("✅ مدل PyG با موفقیت آموزش دید و در فایل‌های 'pyg_tool_model.pth' و 'pyg_data.pt' ذخیره شد.")

if __name__ == "__main__":
    train_and_save()