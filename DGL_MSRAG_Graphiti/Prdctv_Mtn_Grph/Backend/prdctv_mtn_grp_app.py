"""
کد دقیقاً چکار می‌کند؟
این فایل نقطه ورود اصلی (Main Entrypoint) سرویس FastAPI برای سناریوی نگهداری پیش‌بینانه است:
۱. مدیریت لود مدل GNN (ComponentFailureGNN) در حافظه جهت محاسبه ریسک قطعات.
۲. استخراج اطلاعات قطعه از پایگاه دانش GraphRAG و ارسال خروجی GNN به حافظه زمان‌مند Graphiti.
۳. مدیریت سرویس REST API روی پورت 8004 جهت ارایه خروجی چهارلایه (GraphRAG + Graphiti + DGL GNN).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# ۱. فراخوانی معماری به‌روزرسانی‌شده GNN و دیتابیس GraphRAG
from prdctv_mtn_grp_gnn import ComponentFailureGNN, create_sample_industrial_system
from prdctv_mtn_grp_rag import KNOWLEDGE_GRAPH_DB

app = FastAPI(
    title="Industrial Predictive Maintenance Central Engine (GraphRAG + Graphiti + DGL)",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل مستندات نگهداری، حافظه زمان‌مند زنده و پیش‌بینی محاسباتی ریسک خرابی"
)

# ۲. بارگذاری اولیه مدل GNN در حافظه RAM
gnn_model = ComponentFailureGNN(in_dim=3, hidden_dim=16)

# آدرس سرویس API حافظه پویای Graphiti
GRAPHITI_API_URL = "http://127.0.0.1:8004"

class EquipmentQuery(BaseModel):
    query: str

def sync_with_graphiti_memory(equipment_id: str, risk_value: float) -> dict:
    """
    ارسال وضعیت محاسباتی GNN به Graphiti و دریافت آخرین وضعیت زنده فعال
    """
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    
    # ثبت رویداد جدید و منقضی‌سازی وضعیت‌های قبلی در Graphiti
    try:
        requests.post(
            f"{GRAPHITI_API_URL}/add_episode",
            json={
                "source": equipment_id,
                "relation": "has_status",
                "target": f"{status_label} (Risk: {risk_value:.1f}%)"
            },
            timeout=2
        )
    except Exception as e:
        print(f"⚠️ عدم دسترسی به API Graphiti (اجرای حالت رزرو): {e}")

    # بازیابی فکت‌های فعال از Graphiti
    try:
        res = requests.post(
            f"{GRAPHITI_API_URL}/get_active_memory",
            json={"subject": equipment_id},
            timeout=2
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    return {
        "subject": equipment_id,
        "active_facts": [f"{equipment_id} has_status {status_label} (Risk: {risk_value:.1f}%)"],
        "total_episodes_in_history": 1
    }

@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    """
    تحلیل یکپارچه سلامت قطعه صنعتی بر پایه GraphRAG + DGL + Graphiti
    """
    query_text = req.query.lower()
    matched_equipment = None
    
    # گام اول (GraphRAG): استخراج کاندیدا بر اساس پایگاه دانش متنی/کاتالوگ‌ها
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_equipment = data
            break
            
    if not matched_equipment:
        raise HTTPException(
            status_code=404, 
            detail="تجهیز مورد نظر در گراف دانش کاتالوگ‌ها یافت نشد."
        )
    
    equipment_id = matched_equipment["equipment_id"]
    
    # گام دوم (DGL GNN): ساخت گراف و محاسبات عددی ریسک خرابی به ازای هر قطعه
    industrial_graph, sensor_features, component_names = create_sample_industrial_system()
    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # استخراج نمره ریسک اختصاصی قطعه
    equipment_risk = 0.0
    if equipment_id in component_names:
        idx = component_names.index(equipment_id)
        equipment_risk = node_risks[idx].item()
    
    # گام سوم (Graphiti): به‌روزرسانی و بازیابی حافظه زمان‌مند زنده
    graphiti_memory = sync_with_graphiti_memory(equipment_id, equipment_risk)
    
    # گام چهارم: بازگرداندن پاسخ هیبرید کامل
    return {
        "query": req.query,
        "extracted_equipment_id": equipment_id,
        "equipment_type": matched_equipment["equipment_type"],
        "graphrag_maintenance_manual": matched_equipment["description"],
        "graphiti_dynamic_active_facts": graphiti_memory["active_facts"],
        "graphiti_total_history_count": graphiti_memory["total_episodes_in_history"],
        "dgl_predicted_failure_risk_pct": round(equipment_risk, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)