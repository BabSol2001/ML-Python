"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
این فایل نقطه ورود اصلی (Central Entrypoint) سرویس FastAPI است:
۱. استخراج قطعه صنعتی و کاتالوگ تعمیراتی بر اساس ورودی متنی کاربر (GraphRAG).
۲. استعلام زنده ساختار خط تولید و ویژگی سنسورها از پایگاه داده Neo4j GDS.
۳. اجرای محاسبات پیش‌بینی درصد ریسک خرابی قطعات توسط شبکه عصبی گرافی DGL.
۴. ثبت وضعیت جدید قطعه در موتور حافظه زمان‌مند Graphiti و بازیابی فکت‌های فعال.
۵. ارایه پاسخ سه‌گانه/چهارلایه یکپارچه در قالب REST API روی پورت 8008.
========================================================================================
"""

import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ۱. فراخوانی معماری به‌روزرسانی‌شده GNN و توابع اتصال به Neo4j GDS
from prdctv_mtn_grp_neo_gnn import ComponentFailureGNN, fetch_graph_from_neo4j_gds
from prdctv_mtn_grp_neo_rag import KNOWLEDGE_GRAPH_DB


# ========================================================================================
# موتور حافظه زمان‌مند Graphiti (In-Memory Temporal Graph Memory Engine)
# ========================================================================================
class GraphitiMemoryEngine:
    def __init__(self):
        # نگهداری سابقه رویدادها به ازای هر تجهیز
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}

    def add_episode(self, source: str, relation: str, target: str) -> None:
        if source not in self._episodes:
            self._episodes[source] = []
        
        episode = {
            "source": source,
            "relation": relation,
            "target": target,
            "status": "active"
        }
        
        # منقضی‌سازی وضعیت‌های قبلی (Invalidation)
        for prev_ep in self._episodes[source]:
            prev_ep["status"] = "invalidated"
            
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
# مقداردهی اولیه سرویس‌ها
# ========================================================================================
app = FastAPI(
    title="Industrial Predictive Maintenance Central Engine (Neo4j GDS + DGL + Graphiti + GraphRAG)",
    description="سرویس مرکزی یکپارچه‌ساز تحلیل مستندات نگهداری، حافظه زمان‌مند زنده، گراف صنعتی Neo4j و پیش‌بینی GNN"
)

# ۲. بارگذاری اولیه مدل GNN در حافظه RAM (۳ ویژگی ورودی سنسور، ۱۶ نورون مخفی)
gnn_model = ComponentFailureGNN(in_dim=3, hidden_dim=16)

# ساخت نمونه زنده از موتور حافظه Graphiti
graphiti_engine = GraphitiMemoryEngine()

# آدرس‌های اتصال به دیتابیس Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


class EquipmentQuery(BaseModel):
    query: str


def sync_with_graphiti_memory(equipment_id: str, risk_value: float) -> Dict[str, Any]:
    """
    ارسال وضعیت محاسباتی GNN به موتور Graphiti و دریافت وضعیت زنده به همراه تاریخچه
    """
    status_label = "Critical" if risk_value > 70.0 else "Normal"
    
    # ثبت رویداد جدید و منقضی‌سازی اتوماتیک فکت‌های قبلی
    graphiti_engine.add_episode(
        source=equipment_id,
        relation="has_status",
        target=f"{status_label} (Risk: {risk_value:.1f}%)"
    )

    # بازیابی حافظه فعال و تعداد کل اپیزودها
    return graphiti_engine.get_active_memory(equipment_id)


@app.post("/analyze_equipment")
def analyze_equipment(req: EquipmentQuery):
    """
    تحلیل یکپارچه سلامت قطعه صنعتی بر پایه GraphRAG + Neo4j GDS + DGL + Graphiti
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
    
    # گام دوم (Neo4j GDS + DGL GNN): دریافت گراف از Neo4j (با پشتیبانی Auto-Seeding) و اجرای محاسبات ریسک
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
    node_risks = gnn_model(industrial_graph, sensor_features)
    
    # استخراج نمره ریسک اختصاصی قطعه
    equipment_risk = 0.0
    if equipment_id in component_names:
        idx = component_names.index(equipment_id)
        equipment_risk = node_risks[idx].item()
    else:
        equipment_risk = node_risks.mean().item()
    
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
    uvicorn.run(app, host="0.0.0.0", port=8008)