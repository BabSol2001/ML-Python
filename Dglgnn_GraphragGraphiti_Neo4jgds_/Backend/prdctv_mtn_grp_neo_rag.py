"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
سرویس اصلی FastAPI جهت ارکستراسیون سیستم ۵ لایه‌ای:
۱. رویداد Startup: ساخت اتوماتیک گراف و ۱۰۰ نود در Neo4j GDS به محض راه‌اندازی کانتینر.
۲. GraphRAG: استخراج هوشمند و دقیق شماره قطعه صنعتی از متن ورودی (Regex + Keyword).
۳. Neo4j GDS: فراخوانی ویژگی‌های ساختاری ۱۰۰ نود (۷ بعدی) و اصلاح Self-loop جهت جلوگیری از 0-in-degree.
۴. DGL GNN: محاسبه عددی درصد ریسک خرابی قطعه.
۵. Graphiti External API: ثبت سابقه و استعلام حافظه زمان‌مند قطعه از سرویس جانبی (با Fallback ایمن).
========================================================================================
"""

import os
import re
import dgl
import requests
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase

# فراخوانی معماری GNN و توابع استعلام/ساخت دیتابیس از Neo4j GDS
from prdctv_mtn_grp_neo_gnn import (
    ComponentFailureGNN, 
    fetch_graph_from_neo4j_gds, 
    init_database_on_startup
)

app = FastAPI(
    title="Industrial Predictive Maintenance Engine (GraphRAG + Graphiti + Neo4j GDS + DGL)",
    description="سرویس یکپارچه پایش هوشمند تجهیزات صنعتی شامل مستندات فنی، حافظه زمان‌مند، دیتابیس گراف صنعتی و پیش‌بینی GNN"
)

# ۱. بارگذاری مدل GNN (ورودی ۷ بعدی: ۳ سنسور + ۲ FastRP + ۱ PageRank + ۱ Louvain)
gnn_model = ComponentFailureGNN(in_dim=7, hidden_dim=16)

# آدرس سرویس‌های جانبی و دیتابیس Neo4j (سازگار با محیط کانتینری داکر)
GRAPHITI_API_URL = os.getenv("GRAPHITI_API_URL", "http://predictive-maintenance-ai:8008")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-gds:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


# 🚀 ساخت اتوماتیک ۱۰۰ نود و اتصالات دیتابیس به محض Startup کانتینر
@app.on_event("startup")
def auto_initialize_neo4j_graph():
    print("⏳ در حال راه‌اندازی اتوماتیک دیتابیس Neo4j و پروژکشن GDS...")
    try:
        init_database_on_startup()
        print("🚀 گراف Neo4j و ۱۰۰ نود صنعتی با موفقیت به صورت اتوماتیک ساخته شدند.")
    except Exception as e:
        print(f"⚠️ خطای ساخت اتوماتیک گراف در Startup: {e}")


class EquipmentQuery(BaseModel):
    query: str


def fetch_topology_neighbors_from_neo4j(equipment_id: str) -> List[str]:
    """استعلام مستقیم گره‌های متصل در خط تولید از Neo4j"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            query = """
            MATCH (e:Equipment {id: $eq_id})-[r:CONNECTED_TO]-(neighbor:Equipment)
            RETURN DISTINCT neighbor.id AS neighbor_id
            """
            result = session.run(query, eq_id=equipment_id)
            neighbors = [record["neighbor_id"] for record in result]
            driver.close()
            return neighbors
    except Exception as e:
        print(f"⚠️ خطای استعلام توپولوژی از Neo4j ({e}). استفاده از لیست خالی...")
        return []


def update_and_get_graphiti_memory(equipment_id: str, risk_value: float) -> Dict[str, Any]:
    """ثبت وضعیت محاسبه‌شده توسط GNN در Graphiti و استعلام آخرین فکت‌های معتبر"""
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    
    # ارسال فکت جدید به Graphiti اکسترنال
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

    # استعلام آخرین فکت‌های زنده از Graphiti اکسترنال
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
    """اندپوینت اصلی جهت آنالیز یکپارچه هیبرید"""
    query_text = req.query.upper()
    
    # گام اول (Neo4j GDS + DGL GNN): بارگذاری گراف و اجرای محاسبات ریسک GNN برای ۱۰۰ نود بر اساس ۷ ویژگی
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
    
    # 🛠️ اصلاح Self-loop برای جلوگیری از خطای 0-in-degree در DGL GNN
    try:
        industrial_graph = dgl.add_self_loop(industrial_graph)  # type: ignore
    except AttributeError:
        industrial_graph = industrial_graph.add_self_loops()  # type: ignore

    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # گام دوم (GraphRAG Dynamic Search): استخراج و تطبیق دقیق شماره و اسم تجهیز
    matched_equipment_id = None
    
    # ۱. تطبیق مستقیم اسامی کامل (مانند PUMP-4 یا VALVE-8)
    for comp_name in component_names:
        clean_comp = comp_name.upper()
        clean_no_dash = clean_comp.replace("-", "")
        if clean_comp in query_text or clean_no_dash in query_text:
            matched_equipment_id = comp_name
            break

    # ۲. استخراج هوشمند بر اساس اعداد متن (مثلاً استخراج ۴ از "پمپ ۴")
    if not matched_equipment_id:
        numbers = re.findall(r'\d+', query_text)
        if numbers:
            target_num = int(numbers[0])
            if "پمپ" in req.query or "PUMP" in query_text:
                matched_equipment_id = f"PUMP-{target_num}"
            elif "والو" in req.query or "شیر" in req.query or "VALVE" in query_text:
                matched_equipment_id = f"VALVE-{target_num}"
            elif "موتور" in req.query or "MOTOR" in query_text:
                matched_equipment_id = f"MOTOR-{target_num}"
            else:
                matched_equipment_id = next((c for c in component_names if c.endswith(f"-{target_num}")), None)

    # ۳. Fallback ایمن به اولین نود در صورت عدم تطبیق
    if not matched_equipment_id or matched_equipment_id not in component_names:
        matched_equipment_id = component_names[0]

    # استخراج ریسک عددی قطعه
    idx = component_names.index(matched_equipment_id)
    equipment_risk = node_risks[idx].item()
    
    # گام سوم (Neo4j Topology): استخراج اتصالات فیزیکی خط تولید
    connected_neighbors = fetch_topology_neighbors_from_neo4j(matched_equipment_id)
    
    # گام چهارم (Graphiti): به‌روزرسانی و دریافت حافظه زمان‌مند
    graphiti_memory = update_and_get_graphiti_memory(matched_equipment_id, equipment_risk)
    
    # تولید راهنمای فنی دینامیک (GraphRAG Manual)
    manual_desc = f"دستورالعمل تعمیرات پیشگیرانه برای تجهیز {matched_equipment_id}: بررسی دوره‌ای ارتعاش و روانکاری."

    # گام پنجم: تولید خروجی یکپارچه
    return {
        "query": req.query,
        "extracted_equipment_id": matched_equipment_id,
        "graphrag_static_manual": manual_desc,
        "neo4j_connected_topology_components": connected_neighbors,
        "graphiti_dynamic_active_facts": graphiti_memory["active_facts"],
        "graphiti_history_episodes_count": graphiti_memory["total_episodes_in_history"],
        "dgl_gnn_predicted_risk_pct": round(equipment_risk, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)