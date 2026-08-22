"""
کد دقیقا چکار می‌کند؟
این اسکریپت یک شبکه عصبی گرافی (GCN) می‌سازد که ساختار کریستالی مواد را تحلیل کرده و مقدار Bandgap (گاف انرژی) آن‌ها را تخمین می‌زند:
- نمایش اتم‌ها به عنوان گراف: هر اتم یک گره (Node) و اتصالات شیمیایی بین اتم‌ها یا‌ل‌ها (Edges) هستند.
- بردار ویژگی اتم‌ها (Node Features): اطلاعات فیزیکی اتم (مانند عدد اتمی، شعاع اتمی و الکترونگاتیوی) به بردار عددی تبدیل می‌شود.
- پردازش با GNN (لایه GraphConv): شبکه عصبی ویژگی‌های هر اتم را با اتم‌های همسایه‌اش ترکیب کرده و الگوی ساختاری کل کریستال را درک می‌کند.
- پیش‌بینی Bandgap: در نهایت، ویژگی‌های کل اتم‌های کریستال با هم میانگین‌گیری شده (Readout Phase) و یک عدد برحسب الکترون‌ولت (eV) به عنوان Bandgap پیش‌بینی می‌شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
# استفاده از این روش import برای رفع خطای PyLance
import dgl.nn.pytorch as dglnn

# ۱. تعریف معماری شبکه عصبی گرافی برای کریستال‌ها
class CrystalGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super(CrystalGNN, self).__init__()
        # لایه‌های کانوولوشن گرافی (GCN) برای انتقال پیام بین اتم‌ها
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim) # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim) # type: ignore
        # لایه خطی برای تبدیل خروجی به عدد Bandgap
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        # مرحله اول: استخراج ویژگی از همسایگی اتم‌ها
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        
        # مرحله دوم: میانگین‌گیری از کل اتم‌ها (Readout Phase)
        g.ndata['h'] = h
        bg = dgl.mean_nodes(g, 'h')
        
        # مرحله سوم: پیش‌بینی مقدار Bandgap (برحسب eV)
        bandgap = self.regressor(bg)
        return bandgap

# ۲. تابع ساخت ساختار کریستالی (به عنوان ورودی مدل)
def create_sample_crystal():
    src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    
    # جایگزینی مستقیم و بدون خطا
    g = dgl.heterograph({('_N', '_E', '_N'): (src, dst)}) # type: ignore
    
    # ویژگی اتم‌ها: [عدد اتمی نورمال‌شده، شعاع اتمی، الکترونگاتیوی]
    features = torch.tensor([
        [0.40, 0.75, 0.82], # اتم مرکزی (مثلاً Titanium/Lead)
        [0.16, 0.40, 0.89], # اتم اکسیژن ۱
        [0.16, 0.40, 0.89], # اتم اکسیژن ۲
        [0.16, 0.40, 0.89], # اتم اکسیژن ۳
        [0.16, 0.40, 0.89]  # اتم اکسیژن ۴
    ], dtype=torch.float32)
    
    return g, features

# ۳. تست اجرای مدل (برای اطمینان از صحت محاسبات)
if __name__ == "__main__":
    crystal_graph, atom_features = create_sample_crystal()
    
    # ورودی: ۳ ویژگی برای هر اتم | لایه مخفی: ۱۶ نورون
    model = CrystalGNN(in_dim=3, hidden_dim=16)
    
    # محاسبه پیش‌بینی
    predicted_bandgap = model(crystal_graph, atom_features)
    
    print(f"✅ گراف ساختار کریستالی با موفقیت ساخته شد.")
    print(f"📊 پیش‌بینی اولیه GNN برای Bandgap این ماده: {predicted_bandgap.item():.2f} eV")