"""
کد دقیقاً چکار می‌کند؟
این اسکریپت یک شبکه عصبی گرافی (GCN) می‌سازد که ساختار هندسی شبکه لوله‌کشی و سیالات را تحلیل کرده و افت فشار (Pressure Drop) سیستم را تخمین می‌زند:
- نمایش شبکه سیالات به عنوان گراف: هر گره (Node) یک اتصال/نازل/مخزن و هر یال (Edge) مسیر لوله‌کشی است.
- بردار ویژگی گره‌ها (Node Features): پارامترهای هیدرولیکی (نرخ جریان/دبی، فشار ورودی، ضریب اصطکاک لوله) به بردار عددی تبدیل می‌شوند.
- پردازش با GNN (لایه GraphConv): شبکه عصبی ویژگی‌های جریان را بین گره‌های همسایه منتقل کرده و افت فشار کل شبکه را درک می‌کند.
- پیش‌بینی افت فشار: در نهایت، ویژگی‌های کل شبکه با هم میانگین‌گیری شده (Readout Phase) و یک عدد برحسب بار (Bar) پیش‌بینی می‌شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
# استفاده از این روش import برای رفع خطای PyLance
import dgl.nn.pytorch as dglnn

# ۱. تعریف معماری شبکه عصبی گرافی برای سیستم‌های انرژی و سیالات
class FluidEnergyGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super(FluidEnergyGNN, self).__init__()
        # لایه‌های کانوولوشن گرافی (GCN) برای انتقال جریان و محاسبات فیزیکی
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim) # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim) # type: ignore
        # لایه خطی برای تبدیل خروجی به مقدار افت فشار (Bar)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        # مرحله اول: استخراج ویژگی از همسایگی گره‌های لوله‌کشی
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        
        # مرحله دوم: میانگین‌گیری از کل شبکه سیال (Readout Phase)
        g.ndata['h'] = h
        bg = dgl.mean_nodes(g, 'h')
        
        # مرحله سوم: پیش‌بینی میزان افت فشار کل سیستم (برحسب Bar)
        pressure_drop = self.regressor(bg)
        return pressure_drop

# ۲. تابع ساخت گراف شبکه لوله‌کشی (به عنوان ورودی مدل)
def create_sample_fluid_network():
    # اتصال منبع اصلی (۰: ورودی پمپ سیال) به بخش‌های مختلف (۱: زانویی اول، ۲: نازل فشار، ۳: انشعاب خنک‌کننده، ۴: خروجی)
    src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    
    # ساخت گراف ناهمگن مطابق ساختار DGL
    g = dgl.heterograph({('_N', '_E', '_N'): (src, dst)}) # type: ignore
    
    # ویژگی گره‌های لوله‌کشی: [نرخ جریان (دبی)، فشار ورودی، ضریب اصطکاک/زبری لوله]
    features = torch.tensor([
        [0.80, 0.90, 0.15], # گره ورودی اصلی (دبی و فشار بالا)
        [0.60, 0.70, 0.25], # زانویی اول
        [0.40, 0.85, 0.30], # نازل فشار
        [0.55, 0.60, 0.20], # انشعاب خنک‌کننده
        [0.30, 0.40, 0.10]  # گره خروجی
    ], dtype=torch.float32)
    
    return g, features

# ۳. تست اجرای مدل (برای اطمینان از صحت محاسبات)
if __name__ == "__main__":
    fluid_graph, hydraulic_features = create_sample_fluid_network()
    
    # ورودی: ۳ ویژگی هیدرولیکی برای هر گره | لایه مخفی: ۱۶ نورون
    model = FluidEnergyGNN(in_dim=3, hidden_dim=16)
    
    # محاسبه پیش‌بینی
    predicted_pressure_drop = model(fluid_graph, hydraulic_features)
    
    print(f"✅ گراف شبکه سیالات با موفقیت ساخته شد.")
    print(f"📊 پیش‌بینی اولیه GNN برای افت فشار سیستم: {predicted_pressure_drop.item():.2f} Bar")