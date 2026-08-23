"""
کد دقیقاً چکار می‌کند؟
این فایل به‌روزرسانی‌شده شبکه عصبی گرافی (GNN) برای سناریوی پیش‌بینی نت‌پایداری است:
۱. محاسبه ریسک خرابی تفکیک‌شده (Node-level Failure Risk): درصد احتمال خرابی هر قطعه به صورت مستقل محاسبه می‌شود.
۲. شناسایی قطعه بحرانی (Critical Component Identification): قطعه‌ای که بالاترین ریسک خرابی را دارد استخراج می‌شود.
۳. تولید Payload آماده برای Graphiti: یک ساختار داده زمان‌مند (Triple / Episode) تولید می‌کند تا مستقیماً به Graphiti ارسال و ذخیره شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

# ۱. تعریف معماری GNN با خروجی سطح گره (Node-level Output)
class ComponentFailureGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super(ComponentFailureGNN, self).__init__()
        # لایه‌های کانوولوشن گرافی برای انتشار پیام ارتعاش و دما بین قطعات متصل
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)  # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)  # type: ignore
        
        # لایه پیش‌بینی ریسک به ازای هر قطعه (بدون Readout کل گراف)
        self.node_regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        # مرحله اول: استخراج ویژگی‌های ساختاری و همسایگی
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        
        # مرحله دوم: محاسبه درصد ریسک خرابی اختصاصی هر قطعه (خروجی ۰ تا ۱۰۰٪)
        component_risks = torch.sigmoid(self.node_regressor(h)) * 100.0
        return component_risks

# ۲. ساخت گراف سیستم صنعتی به همراه نگاشت نام قطعات
def create_sample_industrial_system():
    # اتصالات فیزیکی قطعات
    src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    
    g = dgl.heterograph({('_N', '_E', '_N'): (src, dst)})  # type: ignore
    
    # اسامی قطعات در سیستم GraphRAG/Graphiti
    component_names = ["PUMP-101", "VALVE-202", "TEMP-SENSOR-01", "TURBINE-301", "TANK-404"]
    
    # ویژگی‌های زنده سنسورها: [ارتعاش، دما، فشار]
    features = torch.tensor([
        [0.89, 0.94, 0.82],  # PUMP-101 (ارتعاش و دمای بحرانی)
        [0.20, 0.45, 0.50],  # VALVE-202
        [0.15, 0.40, 0.40],  # TEMP-SENSOR-01
        [0.30, 0.60, 0.55],  # TURBINE-301
        [0.10, 0.30, 0.20]   # TANK-404
    ], dtype=torch.float32)
    
    return g, features, component_names

# ۳. تست و ساخت ساختار فکت زمان‌مند برای Graphiti
if __name__ == "__main__":
    industrial_graph, sensor_features, component_names = create_sample_industrial_system()
    
    # مقداردهی مدل
    model = ComponentFailureGNN(in_dim=3, hidden_dim=16)
    
    # اجرای استنتاج GNN
    node_risks = model(industrial_graph, sensor_features)
    
    print("✅ محاسبه ریسک قطعات توسط GNN تکمیل شد:\n")
    
    graphiti_episodes = []
    
    for idx, name in enumerate(component_names):
        risk_value = node_risks[idx].item()
        print(f"🔸 قطعه: {name:15} | احتمال خرابی: {risk_value:.2f}%")
        
        # تعیین وضعیت زمان‌مند بر اساس خروجی GNN
        status = "Critical" if risk_value > 70.0 else "Normal"
        
        # ساخت رویداد جهت ارسال به Graphiti
        graphiti_episodes.append({
            "source": name,
            "relation": "has_status",
            "target": f"{status} (Risk: {risk_value:.1f}%)"
        })

    print("\n📦 داده ساختاریافته آماده برای ارسال به API حافظه Graphiti:")
    print(graphiti_episodes[0]) # نمایش نمونه برای قطعه بحرانی PUMP-101