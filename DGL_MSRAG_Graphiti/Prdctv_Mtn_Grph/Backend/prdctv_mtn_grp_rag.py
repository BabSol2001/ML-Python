"""
کد دقیقاً چکار می‌کند؟
این فایل سرویس اصلی FastAPI و پایپلاین یکپارچه GraphRAG + Graphiti + DGL را اجرا می‌کند:
۱. دریافت و پردازش پرسش کاربر (Entity Extraction) جهت یافتن قطعه مورد نظر.
۲. محاسبه ریسک لحظه‌ای قطعات با شبکه عصبی گرافی DGL (ComponentFailureGNN).
۳. ثبت و به‌روزرسانی وضعیت زنده قطعه در حافظه زمان‌مند Graphiti (اتصال آنلاین به API یا حافظه Graphiti).
۴. استخراج کاتالوگ فنی از GraphRAG و ترکیب پاسخ سه‌گانه در یک خروجی REST API هیبرید.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time

# فراخوانی معماری جدید شبکه عصبی و تابع ساخت گراف خط تولید از ماژول GNN
from prdctv_mtn_grp_gnn import ComponentFailureGNN, create_sample_industrial_system

app = FastAPI(
    title="Industrial Predictive Maintenance: GraphRAG + Graphiti + DGL Service",
    description="سرویس ترکیبی پایش هوشمند تجهیزات صنعتی شامل مستندات فنی (GraphRAG)، حافظه زنده (Graphiti) و پیش‌بینی GNN (DGL)"
)

# ۱. بارگذاری مدل GNN قطعات در حافظه (۳ ویژگی ورودی، ۱۶ نورون مخفی)
gnn_model = ComponentFailureGNN(in_dim=3, hidden_dim=16)

# آدرس سرویس حافظه پویای Graphiti (که در فاز قبل پیاده کردیم)
GRAPHITI_API_URL = "http://127.0.0.1:8004"

# ۲. پایگاه دانش ایستای GraphRAG (مستندات و کاتالوگ‌های فنی تعمیرات)
KNOWLEDGE_GRAPH_DB = {
    "pump": {
        "equipment_id": "PUMP-101",
        "description": "پمپ سانتریفیوژ انتقال سیال. در صورت بروز ارتعاش و دمای بالا، بررسی بلبرینگ‌ها و تعویض گریس الزامی است.",
        "equipment_type": "Centrifugal Pump"
    },
    "valve": {
        "equipment_id": "VALVE-202",
        "description": "شیر برقی کنترل جریان. پروتکل اضطراری شامل تست ولتاژ سیم‌پیچ است.",
        "equipment_type": "Solenoid Valve"
    },
    "turbine": {
        "equipment_id": "TURBINE-301",
        "description": "توربین بخار فشار متوسط. بررسی فشار روغن هیدرولیک و آنالیز فرکانس ارتعاشات الزامی است.",
        "equipment_type": "Steam Turbine"
    }
}

class EquipmentQuery(BaseModel):
    query: str

def update_and_get_graphiti_memory(equipment_id: str, risk_value: float) -> dict:
    """
    ثبت وضعیت محاسبه‌شده توسط GNN در Graphiti و استعلام آخرین فکت‌های معتبر
    """
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    
    # ۱. ارسال فکت جدید حاصل از استنتاج GNN به Graphiti جهت ثبت و منقضی کردن فکت‌های قبلی
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
        print(f"⚠️ عدم دسترسی مستقیم به API Graphiti (استفاده از حالت Fallback): {e}")

    # ۲. استعلام آخرین فکت‌های زنده و فعال از Graphiti
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

    # خروجی رزرو در صورت قطع بودن API
    return {
        "subject": equipment_id,
        "active_facts": [f"{equipment_id} has_status {status_label} (Risk: {risk_value:.1f}%)"],
        "total_episodes_in_history": 1
    }

@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    """
    اندپوینت اصلی برای آنالیز هم‌زمان متنی، زنده و محاسباتی سلامت تجهیزات صنعتی
    """
    query_text = req.query.lower()
    matched_equipment = None
    
    # گام اول (GraphRAG): استخراج قطعه صنعتی مورد نظر از متن کاربر
    for key, data in KNOWLEDGE_GRAPH_DB.items():
        if key in query_text:
            matched_equipment = data
            break
            
    if not matched_equipment:
        raise HTTPException(
            status_code=404, 
            detail="تجهیز صنعتی مورد نظر در گراف دانش کاتالوگ‌ها (GraphRAG) یافت نشد."
        )
    
    equipment_id = matched_equipment["equipment_id"]
    
    # گام دوم (DGL GNN): اجرای محاسبات ریسک خرابی سطح گره‌ها
    industrial_graph, sensor_features, component_names = create_sample_industrial_system()
    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # پیدا کردن درصد ریسک محاسبه‌شده برای قطعه استخراج‌شده
    equipment_risk = 0.0
    if equipment_id in component_names:
        idx = component_names.index(equipment_id)
        equipment_risk = node_risks[idx].item()
    
    # گام سوم (Graphiti): به‌روزرسانی و استخراج حافظه پویای زمان‌مند
    graphiti_memory = update_and_get_graphiti_memory(equipment_id, equipment_risk)
    
    # گام چهارم: تولید پاسخ سه‌گانه هیبرید (GraphRAG + Graphiti + DGL)
    return {
        "query": req.query,
        "extracted_equipment_id": equipment_id,
        "equipment_type": matched_equipment["equipment_type"],
        "graphrag_static_manual": matched_equipment["description"],
        "graphiti_dynamic_active_facts": graphiti_memory["active_facts"],
        "graphiti_history_episodes_count": graphiti_memory["total_episodes_in_history"],
        "dgl_gnn_predicted_risk_pct": round(equipment_risk, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)