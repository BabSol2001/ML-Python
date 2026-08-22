"""
کد دقیقاً چکار می‌کند؟
این فایل سرویس اصلی FastAPI و پایپلاین یکپارچه GraphRAG + DGL را برای پایش تجهیزات صنعتی اجرا می‌کند:
- نگهداری پایگاه دانش (GraphRAG KB): کاتالوگ‌های فنی، راهنماهای تعمیرات (Manuals) و پروتکل‌های نگهداری را ذخیره می‌کند.
- استخراج هوشمند قطعه (Entity Extraction): متن پرسش کاربر را آنالیز کرده و تجهیز صنعتی مرتبط را شناسایی می‌کند.
- شبیه‌سازی و پیش‌بینی ریسک (DGL Engine): ساختار اتصالات فیزیکی قطعات را به صورت گراف زنده ساخته و احتمال خرابی (Failure Risk %) را با مدل GNN محاسبه می‌کند.
- ارائه خروجی هیبرید (Hybrid Output): دستورالعمل متنی استخراج‌شده از GraphRAG را با خروجی عددی DGL ترکیب کرده و به صورت REST API برمی‌گرداند.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# فراخوانی معماری شبکه عصبی و تابع ساخت گراف خط تولید از ماژول جدید DGL
from prdctv_mtn_gnn import PredictiveMtnGNN, create_sample_industrial_system

# تعریف سرویس API
app = FastAPI(
    title="Industrial Predictive Maintenance GraphRAG & DGL Service",
    description="سرویس ترکیبی استخراج مستندات فنی تعمیرات و پیش‌بینی محاسباتی ریسک خرابی تجهیزات صنعتی"
)

# ۱. بارگذاری اولیه مدل DGL در حافظه (ورودی ۳ ویژگی سنسوری، ۱۶ نورون لایه مخفی)
gnn_model = PredictiveMtnGNN(in_dim=3, hidden_dim=16)

# ۲. دیتابیس فرضی مستندات فنی و گراف دانش نگهداری صنعتی استخراج‌شده توسط GraphRAG
KNOWLEDGE_GRAPH_DB = {
    "pump": {
        "equipment_id": "PUMP-101",
        "description": "پمپ سانتریفیوژ انتقال سیال. در صورت بروز ارتعاش بالا، بررسی بلبرینگ‌ها و گریس‌کاری مجدد الزامی است.",
        "nodes": ["Pump_Body", "Solenoid_Valve", "Temp_Sensor"],
        "equipment_type": "Centrifugal Pump"
    },
    "turbine": {
        "equipment_id": "TURB-202",
        "description": "توربین بخار فشار متوسط. پروتکل اضطراری شامل کاهش بار ورودی و بررسی فشار روغن هیدرولیک می‌باشد.",
        "nodes": ["Turbine_Rotor", "Pressure_Valve", "Cooling_Tank"],
        "equipment_type": "Steam Turbine"
    }
}

# مدل ورودی درخواست API
class EquipmentQuery(BaseModel):
    query: str

@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    """
    اندپوینت اصلی برای تحلیل متنی و محاسباتی سلامت تجهیزات صنعتی
    """
    query_text = req.query.lower()
    matched_equipment = None
    
    # گام اول: بخش GraphRAG - جستجو و استخراج هوشمند قطعه صنعتی از متن پرسش
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_equipment = data
            break
            
    if not matched_equipment:
        raise HTTPException(
            status_code=404, 
            detail="تجهیز یا قطعه صنعتی مرتبط در گراف دانش GraphRAG یافت نشد."
        )
    
    # گام دوم: بخش DGL - ساخت گراف خط تولید و محاسبه درصد ریسک خرابی (Failure Risk)
    industrial_graph, sensor_features = create_sample_industrial_system()
    predicted_risk = gnn_model(industrial_graph, sensor_features).item()
    
    # گام سوم: ترکیب و بازگرداندن خروجی هیبرید (دستورالعمل متنی + محاسبات ریسک)
    return {
        "query": req.query,
        "extracted_equipment_id": matched_equipment["equipment_id"],
        "equipment_type": matched_equipment["equipment_type"],
        "graphrag_maintenance_manual": matched_equipment["description"],
        "dgl_predicted_failure_risk_pct": round(predicted_risk, 2)
    }

# اجرای محلی جهت تست دستی
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)