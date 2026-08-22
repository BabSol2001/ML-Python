"""
کد دقیقاً چکار می‌کند؟
این فایل سرویس اصلی FastAPI و پایپلاین یکپارچه GraphRAG + DGL را اجرا می‌کند:
- نگهداری پایگاه دانش (GraphRAG KB): اطلاعات متنی، فرمول‌های شیمیایی، توصیفات و روابط کریستالی حاصل از اسناد علمی را ذخیره می‌کند.
- استخراج هوشمند ماده (Entity Extraction): متن پرسش کاربر را آنالیز کرده و ماده یا ساختار کریستالی مرتبط را شناسایی می‌کند.
- شبیه‌سازی و پیش‌بینی فیزیکی (DGL Engine): ساختار اتصالات اتمی کریستال را به صورت گراف زنده ساخته و گاف انرژی (Bandgap) را با مدل GNN محاسبه می‌کند.
- ارائه خروجی هیبرید (Hybrid Output): پاسخ متنی استخراج‌شده از GraphRAG را با خروجی عددی DGL ترکیب کرده و به صورت REST API برمی‌گرداند.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# فراخوانی معماری شبکه عصبی و تابع ساخت گراف اتصالات کریستال از ماژول DGL
from crystal_gnn import CrystalGNN, create_sample_crystal

# تعریف سرویس API
app = FastAPI(
    title="Solid-State Physics GraphRAG & DGL Service",
    description="سرویس ترکیبی استخراج مفاهیم متنی فیزیک حالت جامد و پیش‌بینی محاسباتی گاف انرژی"
)

# ۱. بارگذاری اولیه مدل DGL در حافظه (ورودی ۳ ویژگی اتمی، ۱۶ نورون لایه مخفی)
gnn_model = CrystalGNN(in_dim=3, hidden_dim=16)

# ۲. دیتابیس فرضی مقالات و گراف دانش استخراج‌شده توسط GraphRAG
KNOWLEDGE_GRAPH_DB = {
    "perovskite": {
        "formula": "BaTiO3",
        "description": "ساختار کریستالی پرووسکایت با پایداری حرارتی بالا و خواص فرواستاتیک.",
        "nodes": ["Barium", "Titanium", "Oxygen"],
        "crystal_type": "Perovskite"
    },
    "oxide": {
        "formula": "TiO2",
        "description": "اکسید تیتانیوم مناسب برای نیمه‌هادی‌های فوتوکاتالیستی و لایه‌های نازک.",
        "nodes": ["Titanium", "Oxygen"],
        "crystal_type": "Rutile"
    }
}

# مدل ورودی درخواست API
class MaterialQuery(BaseModel):
    query: str

@app.post("/analyze_material")
def analyze_material(req: MaterialQuery):
    """
    اندپوینت اصلی برای تحلیل متنی و محاسباتی ماده
    """
    query_text = req.query.lower()
    matched_material = None
    
    # گام اول: بخش GraphRAG - جستجو و استخراج هوشمند ماده از متن پرسش
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_material = data
            break
            
    if not matched_material:
        raise HTTPException(
            status_code=404, 
            detail="ماده یا ساختار کریستالی مرتبط در گراف دانش GraphRAG یافت نشد."
        )
    
    # گام دوم: بخش DGL - ساخت گراف اتمی و محاسبه گاف انرژی (Bandgap)
    crystal_graph, atom_features = create_sample_crystal()
    predicted_bandgap = gnn_model(crystal_graph, atom_features).item()
    
    # گام سوم: ترکیب و بازگرداندن خروجی هیبرید (متن + محاسبات عددی)
    return {
        "query": req.query,
        "extracted_material": matched_material["formula"],
        "crystal_type": matched_material["crystal_type"],
        "graphrag_summary": matched_material["description"],
        "dgl_predicted_bandgap_eV": round(predicted_bandgap, 3)
    }

# اجرای محلی جهت تست دستی
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)