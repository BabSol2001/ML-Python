"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
این فایل نقطه ورود اصلی (Central Entrypoint) سرویس FastAPI است:
۱. ساخت اتوماتیک گراف و ۱۰۰ نود در Neo4j GDS به محض روشن شدن کانتینر (Startup Event).
۲. استخراج هوشمند قطعه از بین ۱۰۰ نود دیتابیس بر اساس ورودی متنی کاربر (GraphRAG Dynamic).
۳. استعلام زنده ساختار خط تولید و ۷ ویژگی (سنسورها + FastRP + PageRank + Louvain) از Neo4j GDS.
۴. اصلاح ساختار گراف جهت حذف نودهای 0-in-degree با افزودن Self-Loop در DGL.
۵. اجرای محاسبات پیش‌بینی درصد ریسک خرابی قطعات توسط شبکه عصبی گرافی DGL.
۶. ثبت وضعیت جدید قطعه در موتور حافظه زمان‌مند درون‌برنامه‌ای (In-Memory Graphiti Engine).
========================================================================================
"""

import os
import re
import dgl
from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase

# فراخوانی معماری GNN و توابع اتصال به Neo4j GDS
from prdctv_mtn_grp_neo_gnn import (
    ComponentFailureGNN, 
    fetch_graph_from_neo4j_gds, 
    init_database_on_startup
)


# ========================================================================================
# موتور حافظه زمان‌مند Graphiti
# ========================================================================================
class GraphitiMemoryEngine:
    def __init__(self):
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}

    def add_episode(self, source: str, relation: str, target: str) -> None:
        if source not in self._episodes:
            self._episodes[source] = []
        
        # منقضی‌سازی تمام وضعیت‌های قبلی این تجهیز
        for prev_ep in self._episodes[source]:
            prev_ep["status"] = "invalidated"
            
        episode = {
            "source": source,
            "relation": relation,
            "target": target,
            "status": "active"
        }
        self._episodes[source].append(episode)

    def get_active_memory(self, subject: str) -> Dict[str, Any]:
        episodes = self._episodes.get(subject, [])
        active_facts = [
            f"{ep['source']} {ep['relation']} {ep['target']}"
            for ep in episodes if ep["status"] == "active"
        ]
        
        return {
            "subject": subject,
            "active_facts": active_facts if active_facts else [f"{subject} has_status Unknown"],
            "total_episodes_in_history": len(episodes)
        }


# ========================================================================================
# مقداردهی اولیه سرویس‌ها و رویداد Startup اتوماتیک
# ========================================================================================
app = FastAPI(
    title="Industrial Predictive Maintenance Central Engine (Neo4j GDS + DGL + Graphiti + GraphRAG)",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل مستندات نگهداری، حافظه زمان‌مند زنده، گراف صنعتی Neo4j و پیش‌بینی GNN"
)

# بارگذاری مدل GNN (ورودی ۷ بعدی)
gnn_model = ComponentFailureGNN(in_dim=7, hidden_dim=16)
graphiti_engine = GraphitiMemoryEngine()

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
    """استعلام مستقیم گره‌های متصل در خط تولید از Neo4j با مدیریت ایمن Driver"""
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            query = """
            MATCH (e:Equipment {id: $eq_id})-[r:CONNECTED_TO]-(neighbor:Equipment)
            RETURN DISTINCT neighbor.id AS neighbor_id
            """
            result = session.run(query, eq_id=equipment_id)
            neighbors = [record["neighbor_id"] for record in result]
            return neighbors
    except Exception as e:
        print(f"⚠️ خطای استعلام توپولوژی از Neo4j ({e}).")
        return []
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass


def sync_with_graphiti_memory(equipment_id: str, risk_value: float) -> Dict[str, Any]:
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    graphiti_engine.add_episode(
        source=equipment_id,
        relation="has_status",
        target=f"{status_label} (Risk: {risk_value:.1f}%)"
    )
    return graphiti_engine.get_active_memory(equipment_id)


@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    query_text = req.query.upper()
    
    # گام اول: استخراج گراف ۷ بعدی از Neo4j GDS
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
    
    # اضافه کردن self-loop برای حل خطای 0-in-degree در DGL
    try:
        industrial_graph = dgl.add_self_loop(industrial_graph)  # type: ignore
    except AttributeError:
        industrial_graph = industrial_graph.add_self_loops()  # type: ignore
    
    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # گام دوم: تطبیق هوشمند شماره تجهیز از متن کاربر (مثلاً PUMP-4)
    matched_equipment_id = None
    
    # ۱. تطبیق مستقیم اسامی موجود
    for comp_name in component_names:
        clean_comp = comp_name.upper()
        if clean_comp in query_text or clean_comp.replace("-", "") in query_text:
            matched_equipment_id = comp_name
            break

    # ۲. تطبیق هوشمند اعداد (مثلاً استخراج عدد ۴ از "PUMP-4" یا "پمپ ۴")
    if not matched_equipment_id:
        numbers = re.findall(r'\d+', query_text)
        if numbers:
            target_num = int(numbers[0])
            matched_equipment_id = next((c for c in component_names if c.endswith(f"-{target_num}")), None)

    # ۳. Fallback بر اساس نوع قطعه
    if not matched_equipment_id:
        if "PUMP" in query_text or "پمپ" in req.query:
            matched_equipment_id = next((c for c in component_names if "PUMP" in c), component_names[0])
        elif "VALVE" in query_text or "والو" in req.query or "شیر" in req.query:
            matched_equipment_id = next((c for c in component_names if "VALVE" in c), component_names[0])
        elif "MOTOR" in query_text or "موتور" in req.query:
            matched_equipment_id = next((c for c in component_names if "MOTOR" in c), component_names[0])
        else:
            matched_equipment_id = component_names[0]

    idx = component_names.index(matched_equipment_id)
    equipment_risk = node_risks[idx].item()
    
    # گام سوم: دریافت اتصالات زنده از دیتابیس
    connected_neighbors = fetch_topology_neighbors_from_neo4j(matched_equipment_id)
    
    # گام چهارم: به‌روزرسانی حافظه Graphiti
    graphiti_memory = sync_with_graphiti_memory(matched_equipment_id, equipment_risk)
    
    manual_desc = f"دستورالعمل نگهداری تجهیز {matched_equipment_id}: تست ارتعاش، دما و بررسی روغن‌کاری دوره‌ای."

    return {
        "query": req.query,
        "extracted_equipment_id": matched_equipment_id,
        "graphrag_maintenance_manual": manual_desc,
        "neo4j_connected_topology_components": connected_neighbors,
        "graphiti_dynamic_active_facts": graphiti_memory["active_facts"],
        "graphiti_total_history_count": graphiti_memory["total_episodes_in_history"],
        "dgl_predicted_failure_risk_pct": round(equipment_risk, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)