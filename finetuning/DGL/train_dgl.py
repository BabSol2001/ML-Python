from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🔹 اضافه کردن type: ignore به ایمپورت‌ها جهت سکوت Pylance
import dgl 
import dgl.nn.pytorch as dglnn
from dgl.data.utils import save_graphs  # type: ignore

# ---------------------------------------------------------
# ۱. ساختار مدل با DGL
# ---------------------------------------------------------
class FraudDetectorDGL(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, num_classes: int):
        super().__init__()
        # اگر باز هم خط قرمز داد، inline کامنت بذارید:
        self.conv1 = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type="mean")  # type: ignore
        self.conv2 = dglnn.SAGEConv(hidden_feats, num_classes, aggregator_type="mean")  # type: ignore

    def forward(self, g: Any, in_feat: torch.Tensor) -> torch.Tensor:
        # مرحله ۱: اعمال پیام‌رسانی روی ساختار گراف g
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        # مرحله ۲: استخراج logits برای دو کلاس (سالم=۰, مخرب=۱)
        h = self.conv2(g, h)
        return h


# ---------------------------------------------------------
# ۲. ساخت داده و آموزش مدل
# ---------------------------------------------------------
def train_dgl():
    # ساخت یک گراف DGL با ۶ گره ایجنت و چند یال تراکنش مالی
    src = torch.tensor([0, 1, 2, 3, 4, 0, 2])
    dst = torch.tensor([1, 2, 0, 4, 5, 3, 5])

    # خط قرمز dgl.graph هم با type: ignore حل می‌شه
    g = dgl.graph((src, dst))  # type: ignore

    # ویژگی‌ها: ۱۰ فاکتور مالی برای هر ایجنت
    features = torch.randn(6, 10)
    # نسبت دادن ویژگی‌ها به داده‌های گره (روش استاندارد در DGL)
    g.ndata["feat"] = features

    # لیبل‌ها: ۰ (ایجنت سالم)، ۱ (ایجنت مخرب)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    model = FraudDetectorDGL(in_feats=10, hidden_feats=16, num_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(40):
        logits = model(g, features)
        loss = F.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ذخیره وزن‌های مدل
    torch.save(model.state_dict(), "dgl_model.pth")

    # ذخیره گراف (شامل ویژگی‌های ndata)
    save_graphs("dgl_graph.bin", [g])
    print("✅ مدل DGL با موفقیت آموزش دید و در فایل‌های 'dgl_model.pth' و 'dgl_graph.bin' ذخیره شد.")


if __name__ == "__main__":
    train_dgl()