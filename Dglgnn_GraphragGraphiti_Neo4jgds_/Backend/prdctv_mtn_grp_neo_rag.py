"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
این فایل سرویس اصلی FastAPI و ارکستریتور یکپارچه ۵ لایه‌ای سیستم است:
۱. GraphRAG: استخراج قطعه صنعتی مورد نظر و کاتالوگ متنی مربوطه از متن ورودی کاربر.
۲. Neo4j GDS: استعلام آنلاین روابط فیزیکی خط تولید و ویژگی‌های سنسورها از دیتابیس Neo4j.
۳. DGL GNN: محاسبه عددی درصد ریسک خرابی قطعه بر اساس ویژگی‌های زنده دیتابیس.
۴. Graphiti: ثبت فکت جدید حاصل از ریسک و استخراج حافظه پویای زمان‌مند قطعه.
۵. FastAPI REST API: ارائه خروجی هیبرید ۵ لایه‌ای روی پورت 8008 برای ایجنت‌ها.
========================================================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Dict, Any, List
from neo4j import GraphDatabase

# فراخوانی معماری جدید GNN و تابع استعلام از Neo4j GDS
from prdctv_mtn_grp_neo_gnn import ComponentFailureGNN, fetch_graph_from_neo4j_gds

app = FastAPI(
    title="Industrial Predictive Maintenance Engine (GraphRAG + Graphiti + Neo4j GDS + DGL)",
    description="سرویس یکپارچه پایش هوشمند تجهیزات صنعتی شامل مستندات فنی، حافظه زمان‌مند، دیتابیس گراف صنعتی و پیش‌بینی GNN"
)

# ۱. بارگذاری مدل GNN قطعات در حافظه (۳ ویژگی ورودی، ۱۶ نورون مخفی)
gnn_model = ComponentFailureGNN(in_dim=3, hidden_dim=16)

# آدرس سرویس‌های جانبی و دیتابیس Neo4j
GRAPHITI_API_URL = "http://127.0.0.1:8008"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

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


def fetch_topology_neighbors_from_neo4j(equipment_id: str) -> List[str]:
    """
    استعلام مستقیم گره‌های متصل در خط تولید از Neo4j
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            query = """
            MATCH (e:Equipment {id: $eq_id})-[:CONNECTED_TO]-(neighbor:Equipment)
            RETURN neighbor.id AS neighbor_id
            """
            result = session.run(query, eq_id=equipment_id)
            neighbors = [record["neighbor_id"] for record in result]
            driver.close()
            return neighbors
    except Exception as e:
        print(f"⚠️ خطای استعلام توپولوژی از Neo4j ({e}). استفاده از لیست خالی...")
        return []


def update_and_get_graphiti_memory(equipment_id: str, risk_value: float) -> Dict[str, Any]:
    """
    ثبت وضعیت محاسبه‌شده توسط GNN در Graphiti و استعلام آخرین فکت‌های معتبر
    """
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    
    # ۱. ارسال فکت جدید به Graphiti
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
        print(f"⚠️ عدم دسترسی به API Graphiti (استفاده از حالت Fallback): {e}")

    # ۲. استعلام آخرین فکت‌های زنده از Graphiti
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
    اندپوینت اصلی جهت آنالیز یکپارچه هیبرید (GraphRAG + Neo4j GDS + DGL GNN + Graphiti)
    """
    query_text = req.query.lower()
    matched_equipment = None
    
    # گام اول (GraphRAG): استخراج قطعه از متن ورودی
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
    
    # گام دوم (Neo4j GDS + DGL GNN): استخراج گراف از Neo4j و اجرای محاسبات ریسک GNN
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # استخراج ریسک عددی قطعه
    equipment_risk = 0.0
    if equipment_id in component_names:
        idx = component_names.index(equipment_id)
        equipment_risk = node_risks[idx].item()
    
    # گام سوم (Neo4j Topology): استخراج اتصالات فیزیکی خط تولید
    connected_neighbors = fetch_topology_neighbors_from_neo4j(equipment_id)
    
    # گام چهارم (Graphiti): به‌روزرسانی و دریافت حافظه زمان‌مند
    graphiti_memory = update_and_get_graphiti_memory(equipment_id, equipment_risk)
    
    # گام پنجم: تولید خروجی یکپارچه هیبرید
    return {
        "query": req.query,
        "extracted_equipment_id": equipment_id,
        "equipment_type": matched_equipment["equipment_type"],
        "graphrag_static_manual": matched_equipment["description"],
        "neo4j_connected_topology_components": connected_neighbors,
        "graphiti_dynamic_active_facts": graphiti_memory["active_facts"],
        "graphiti_history_episodes_count": graphiti_memory["total_episodes_in_history"],
        "dgl_gnn_predicted_risk_pct": round(equipment_risk, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)