"""
کد دقیقاً چکار می‌کند؟
این فایل نقطه ورود اصلی (Main Entrypoint) سرویس FastAPI است:
- یکپارچه‌سازی ماژول‌ها: منطق پردازش متنی GraphRAG (از prdctv_mtn_rag) و محاسبات ریسک GNN (از prdctv_mtn_gnn) را فراخوانی می‌کند.
- مدیریت مدل در حافظه: مدل PredictiveMtnGNN را یک‌بار در زمان لود برنامه مقداردهی اولیه می‌کند تا سرعت پاسخ‌دهی بالا برود.
- مدیریت سرویس REST API: اندپوینت analyze_equipment/ را برای دریافت درخواست‌ها و ارجاع پاسخ ترکیبی فراهم می‌سازد.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ۱. فراخوانی ماژول محاسباتی DGL و ماژول متنی GraphRAG برای نگهداری پیش‌بینانه
from prdctv_mtn_gnn import PredictiveMtnGNN, create_sample_industrial_system
from prdctv_mtn_rag import KNOWLEDGE_GRAPH_DB

app = FastAPI(
    title="Industrial Predictive Maintenance GraphRAG & DGL Engine",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل مستندات نگهداری و پیش‌بینی ریسک خرابی تجهیزات صنعتی"
)

# ۲. بارگذاری و مقداردهی اولیه مدل DGL در حافظه RAM
gnn_model = PredictiveMtnGNN(in_dim=3, hidden_dim=16)

class EquipmentQuery(BaseModel):
    query: str

@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    """
    دریافت پرسش کاربر، استخراج قطعه صنعتی از گراف دانش و ارزیابی محاسباتی ریسک خرابی (Failure Risk)
    """
    query_text = req.query.lower()
    matched_equipment = None
    
    # گام اول: استخراج کاندیدا بر اساس پایگاه دانش متنی
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_equipment = data
            break
            
    if not matched_equipment:
        raise HTTPException(
            status_code=404, 
            detail="تجهیز مورد نظر در گراف دانش یافت نشد."
        )
    
    # گام دوم: ساخت گراف خط تولید و محاسبات عددی با DGL
    industrial_graph, sensor_features = create_sample_industrial_system()
    predicted_risk = gnn_model(industrial_graph, sensor_features).item()
    
    # گام سوم: بازگرداندن خروجی نهایی
    return {
        "query": req.query,
        "extracted_equipment_id": matched_equipment["equipment_id"],
        "equipment_type": matched_equipment["equipment_type"],
        "graphrag_maintenance_manual": matched_equipment["description"],
        "dgl_predicted_failure_risk_pct": round(predicted_risk, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)