"""
کد دقیقاً چکار می‌کند؟
این فایل نقطه ورود اصلی (Main Entrypoint) سرویس FastAPI است:
- یکپارچه‌سازی ماژول‌ها: منطق پردازش متنی GraphRAG (از fld_enrg_rag) و محاسبات افت فشار GNN (از fld_enrg_gnn) را فراخوانی می‌کند.
- مدیریت مدل در حافظه: مدل FluidEnergyGNN را یک‌بار در زمان لود برنامه مقداردهی اولیه می‌کند تا سرعت پاسخ‌دهی بالا برود.
- مدیریت سرویس REST API: اندپوینت analyze_fluid_system/ را برای دریافت درخواست‌ها و ارجاع پاسخ ترکیبی فراهم می‌سازد.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ۱. فراخوانی ماژول محاسباتی DGL و ماژول متنی GraphRAG برای سیستم‌های انرژی و سیالات
from fld_enrg_gnn import FluidEnergyGNN, create_sample_fluid_network
from fld_enrg_rag import KNOWLEDGE_GRAPH_DB

app = FastAPI(
    title="Fluid & Energy Systems GraphRAG & DGL Engine",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل استانداردهای مهندسی سیالات و پیش‌بینی افت فشار شبکه لوله‌کشی"
)

# ۲. بارگذاری و مقداردهی اولیه مدل DGL در حافظه RAM
gnn_model = FluidEnergyGNN(in_dim=3, hidden_dim=16)

class FluidSystemQuery(BaseModel):
    query: str

@app.post("/analyze_fluid_system")
def analyze_fluid_system(req: FluidSystemQuery):
    """
    دریافت پرسش کاربر، استخراج شبکه سیالات از گراف دانش و ارزیابی محاسباتی افت فشار (Pressure Drop)
    """
    query_text = req.query.lower()
    matched_system = None
    
    # گام اول: استخراج کاندیدا بر اساس پایگاه دانش متنی
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_system = data
            break
            
    if not matched_system:
        raise HTTPException(
            status_code=404, 
            detail="سیستم یا شبکه لوله‌کشی مورد نظر در گراف دانش یافت نشد."
        )
    
    # گام دوم: ساخت گراف شبکه سیالات و محاسبات عددی با DGL
    fluid_graph, hydraulic_features = create_sample_fluid_network()
    predicted_pressure_drop = gnn_model(fluid_graph, hydraulic_features).item()
    
    # گام سوم: بازگرداندن خروجی نهایی
    return {
        "query": req.query,
        "extracted_system_id": matched_system["system_id"],
        "standard_code": matched_system["standard_code"],
        "graphrag_engineering_standard": matched_system["description"],
        "dgl_predicted_pressure_drop_bar": round(predicted_pressure_drop, 3)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
