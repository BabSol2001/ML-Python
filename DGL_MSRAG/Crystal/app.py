"""
کد دقیقاً چکار می‌کند؟
این فایل نقطه ورود اصلی (Main Entrypoint) سرویس FastAPI است:
- یکپارچه‌سازی ماژول‌ها: منطق پردازش متنی GraphRAG (از solid_state_rag) و محاسبات فیزیکی GNN (از crystal_gnn) را فراخوانی می‌کند.
- مدیریت مدل در حافظه: مدل CrystalGNN را یک‌بار در زمان لود برنامه مقداردهی اولیه می‌کند تا سرعت پاسخ‌دهی بالا برود.
- مدیریت سرویس REST API: اندپوینت analyze_material/ را برای دریافت درخواست‌ها و ارجاع پاسخ ترکیبی فراهم می‌سازد.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ۱. فراخوانی ماژول محاسباتی DGL و ماژول متنی GraphRAG
from crystal_gnn import CrystalGNN, create_sample_crystal
from solid_state_rag import KNOWLEDGE_GRAPH_DB

app = FastAPI(
    title="Solid-State Physics GraphRAG & DGL Engine",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل متنی مقالات و محاسبات ساختار کریستالی"
)

# ۲. بارگذاری و مقداردهی اولیه مدل DGL در حافظه RAM
gnn_model = CrystalGNN(in_dim=3, hidden_dim=16)

class MaterialQuery(BaseModel):
    query: str

@app.post("/analyze_material")
def analyze_material(req: MaterialQuery):
    """
    دریافت پرسش کاربر، استخراج ماده از گراف دانش و ارزیابی محاسباتی Bandgap
    """
    query_text = req.query.lower()
    matched_material = None
    
    # گام اول: استخراج کاندیدا بر اساس پایگاه دانش متنی
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_material = data
            break
            
    if not matched_material:
        raise HTTPException(
            status_code=404, 
            detail="ماده مورد نظر در گراف دانش یافت نشد."
        )
    
    # گام دوم: ساخت گراف اتصالات و محاسبات عددی با DGL
    crystal_graph, atom_features = create_sample_crystal()
    predicted_bandgap = gnn_model(crystal_graph, atom_features).item()
    
    # گام سوم: بازگرداندن خروجی نهایی
    return {
        "query": req.query,
        "extracted_material": matched_material["formula"],
        "crystal_type": matched_material["crystal_type"],
        "graphrag_summary": matched_material["description"],
        "dgl_predicted_bandgap_eV": round(predicted_bandgap, 3)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)