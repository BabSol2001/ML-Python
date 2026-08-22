"""
کد دقیقاً چکار می‌کند؟
این فایل سرویس اصلی FastAPI و پایپلاین یکپارچه GraphRAG + DGL را برای بهینه‌سازی سیستم‌های انرژی و سیالات اجرا می‌کند:
- نگهداری پایگاه دانش (GraphRAG KB): قوانین مهندسی، استانداردهای ایمنی (ASME B31.3) و محدودیت‌های فشار/دما را ذخیره می‌کند.
- استخراج هوشمند سیستم (Entity Extraction): متن پرسش کاربر را آنالیز کرده و شبکه سیال یا لوله‌کشی مرتبط را شناسایی می‌کند.
- شبیه‌سازی و پیش‌بینی هیدرولیکی (DGL Engine): ساختار شبکه لوله‌کشی را به صورت گراف زنده ساخته و افت فشار (Pressure Drop) را با مدل GNN محاسبه می‌کند.
- ارائه خروجی هیبرید (Hybrid Output): استانداردهای استخراج‌شده از GraphRAG را با خروجی عددی DGL ترکیب کرده و به صورت REST API برمی‌گرداند.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# فراخوانی معماری شبکه عصبی و تابع ساخت گراف شبکه سیالات از ماژول DGL
from fld_enrg_gnn import FluidEnergyGNN, create_sample_fluid_network

# تعریف سرویس API
app = FastAPI(
    title="Fluid & Energy Systems GraphRAG & DGL Service",
    description="سرویس ترکیبی استخراج استانداردهای مهندسی سیالات و پیش‌بینی محاسباتی افت فشار شبکه لوله‌کشی"
)

# ۱. بارگذاری اولیه مدل DGL در حافظه (ورودی ۳ ویژگی هیدرولیکی، ۱۶ نورون لایه مخفی)
gnn_model = FluidEnergyGNN(in_dim=3, hidden_dim=16)

# ۲. دیتابیس فرضی استانداردهای مهندسی و گراف دانش سیالات استخراج‌شده توسط GraphRAG
KNOWLEDGE_GRAPH_DB = {
    "piping": {
        "system_id": "PIPE-SYS-301",
        "description": "مطابق استاندارد ASME B31.3، افت فشار مجاز در این شبکه نباید از ۱.۵ بار تجاوز کند. متریال لوله از فولاد ضدزنگ 316L انتخاب شده است.",
        "nodes": ["Pump_Inlet", "Elbow_Joint", "Pressure_Nozzle"],
        "standard_code": "ASME B31.3"
    },
    "cooling": {
        "system_id": "COOL-NET-402",
        "description": "مطابق استانداردهای ایمنی نیروگاهی، نرخ جریان سیال خنک‌کننده باید در تمام انشعاب‌ها پایدار بماند تا از پدیده کاویتاسیون جلوگیری شود.",
        "nodes": ["Cooling_Branch", "Outlet_Valve"],
        "standard_code": "ISO 5167"
    }
}

# مدل ورودی درخواست API
class FluidSystemQuery(BaseModel):
    query: str

@app.post("/analyze_fluid_system")
def analyze_fluid_system(req: FluidSystemQuery):
    """
    اندپوینت اصلی برای تحلیل متنی استانداردهای مهندسی و محاسبات افت فشار شبکه سیالات
    """
    query_text = req.query.lower()
    matched_system = None
    
    # گام اول: بخش GraphRAG - جستجو و استخراج هوشمند سیستم سیالاتی از متن پرسش
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_system = data
            break
            
    if not matched_system:
        raise HTTPException(
            status_code=404, 
            detail="سیستم یا شبکه لوله‌کشی مرتبط در گراف دانش GraphRAG یافت نشد."
        )
    
    # گام دوم: بخش DGL - ساخت گراف شبکه سیالات و محاسبه افت فشار (Pressure Drop)
    fluid_graph, hydraulic_features = create_sample_fluid_network()
    predicted_pressure_drop = gnn_model(fluid_graph, hydraulic_features).item()
    
    # گام سوم: ترکیب و بازگرداندن خروجی هیبرید (استاندارد متنی + محاسبات افت فشار)
    return {
        "query": req.query,
        "extracted_system_id": matched_system["system_id"],
        "standard_code": matched_system["standard_code"],
        "graphrag_engineering_standard": matched_system["description"],
        "dgl_predicted_pressure_drop_bar": round(predicted_pressure_drop, 3)
    }

# اجرای محلی جهت تست دستی
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)