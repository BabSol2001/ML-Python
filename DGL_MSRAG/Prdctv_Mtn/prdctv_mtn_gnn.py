"""
کد دقیقا چکار می‌کند؟
این اسکریپت یک شبکه عصبی گرافی (GCN) می‌سازد که ساختار فیزیکی تجهیزات صنعتی را تحلیل کرده و احتمال خرابی (Failure Risk Percentage) آن‌ها را تخمین می‌زند:
- نمایش تجهیزات به عنوان گراف: هر قطعه صنعتی (پمپ، شیر، توربین) یک گره (Node) و اتصالات فیزیکی بین آن‌ها یال‌ها (Edges) هستند.
- بردار ویژگی قطعات (Node Features): داده‌های زنده سنسورها (دما، ارتعاش و فشار نرمال‌شده) به بردار عددی تبدیل می‌شوند.
- پردازش با GNN (لایه GraphConv): شبکه عصبی ویژگی‌های سنسوری هر قطعه را با قطعات همسایه‌اش ترکیب کرده و الگوی عملیاتی کل سیستم را درک می‌کند.
- پیش‌بینی احتمال خرابی: در نهایت، ویژگی‌های کل قطعات با هم میانگین‌گیری شده (Readout Phase) و یک عدد بین ۰ تا ۱۰۰ درصد به عنوان ریسک خرابی سیستم پیش‌بینی می‌شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
# استفاده از این روش import برای رفع خطای PyLance
import dgl.nn.pytorch as dglnn

# ۱. تعریف معماری شبکه عصبی گرافی برای تجهیزات صنعتی
class PredictiveMtnGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super(PredictiveMtnGNN, self).__init__()
        # لایه‌های کانوولوشن گرافی (GCN) برای انتقال پیام بین قطعات صنعتی
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim) # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim) # type: ignore
        # لایه خطی برای تبدیل خروجی به درصد ریسک خرابی
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        # مرحله اول: استخراج ویژگی از همسایگی قطعات
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        
        # مرحله دوم: میانگین‌گیری از کل قطعات خط تولید (Readout Phase)
        g.ndata['h'] = h
        bg = dgl.mean_nodes(g, 'h')
        
        # مرحله سوم: پیش‌بینی احتمال خرابی (خروجی بین ۰ تا ۱۰۰ درصد)
        failure_risk = torch.sigmoid(self.regressor(bg)) * 100.0
        return failure_risk

# ۲. تابع ساخت گراف خط تولید (به عنوان ورودی مدل)
def create_sample_industrial_system():
    # اتصال قطعه مرکزی (۰: پمپ اصلی) به سایر قطعات (۱: شیر برقی، ۲: حسگر دما، ۳: توربین، ۴: مخزن)
    src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    
    # ساخت گراف ناهمگن مطابق ساختار DGL
    g = dgl.heterograph({('_N', '_E', '_N'): (src, dst)}) # type: ignore
    
    # ویژگی قطعات صنعتی: [میزان ارتعاش، دما، فشار سیستم]
    features = torch.tensor([
        [0.85, 0.92, 0.78], # قطعه مرکزی (پمپ اصلی - دارای ارتعاش و دمای بالا)
        [0.20, 0.45, 0.50], # شیر برقی ۱
        [0.15, 0.40, 0.40], # حسگر دما
        [0.30, 0.60, 0.55], # توربین
        [0.10, 0.30, 0.20]  # مخزن
    ], dtype=torch.float32)
    
    return g, features

# ۳. تست اجرای مدل (برای اطمینان از صحت محاسبات)
if __name__ == "__main__":
    industrial_graph, sensor_features = create_sample_industrial_system()
    
    # ورودی: ۳ ویژگی سنسوری برای هر قطعه | لایه مخفی: ۱۶ نورون
    model = PredictiveMtnGNN(in_dim=3, hidden_dim=16)
    
    # محاسبه پیش‌بینی
    predicted_risk = model(industrial_graph, sensor_features)
    
    print(f"✅ گراف سیستم صنعتی با موفقیت ساخته شد.")
    print(f"📊 پیش‌بینی اولیه GNN برای احتمال خرابی سیستم: {predicted_risk.item():.2f}%")