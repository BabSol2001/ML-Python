"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
این اسکریپت لایه محاسباتی شبکه‌های عصبی گرافی (DGL GNN) را مدیریت می‌کند:
۱. اتصال به Neo4j GDS: مقداردهی اولیه خودکار (Auto-Seeding) در صورت خالی بودن دیتابیس،
   استخراج اتصالات فیزیکی قطعات و ویژگی‌های سنسورها (ارتعاش، دما، فشار).
۲. نگاشت ساختار Neo4j به DGL: تبدیل گراف استخراج‌شده از دیتابیس به یک شیء HeteroGraph/Homogeneous در DGL.
۳. اجرای مدل GNN اختصاصی (ComponentFailureGNN): محاسبه درصد احتمال خرابی تفکیک‌شده قطعات.
۴. آماده‌سازی خروجی برای Graphiti: ساخت Tripleهای زمان‌مند بر اساس ریسک خرابی جهت ثبت در حافظه.
========================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from neo4j import GraphDatabase
from typing import Tuple, List, Dict, Any


# ۱. تعریف معماری GNN با خروجی سطح گره (Node-level Output)
class ComponentFailureGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super(ComponentFailureGNN, self).__init__()
        # لایه‌های کانوولوشن گرافی برای انتشار پیام ارتعاش و دما بین قطعات متصل
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)  # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)  # type: ignore
        
        # لایه پیش‌بینی ریسک به ازای هر قطعه (بدون Readout کل گراف)
        self.node_regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        # مرحله اول: استخراج ویژگی‌های ساختاری و همسایگی
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        
        # مرحله دوم: محاسبه درصد ریسک خرابی اختصاصی هر قطعه (خروجی ۰ تا ۱۰۰٪)
        component_risks = torch.sigmoid(self.node_regressor(h)) * 100.0
        return component_risks


# تابع کمکی جهت ثبت اتوماتیک گراف پایه در دیتابیس Neo4j
def _auto_seed_neo4j_graph(session):
    seed_nodes_query = """
    UNWIND [
      {id: 'PUMP-101', type: 'Centrifugal Pump', vibration: 0.89, temp: 0.94, pressure: 0.82},
      {id: 'VALVE-202', type: 'Solenoid Valve', vibration: 0.20, temp: 0.45, pressure: 0.50},
      {id: 'TURBINE-301', type: 'Steam Turbine', vibration: 0.30, temp: 0.60, pressure: 0.55}
    ] AS eq
    MERGE (e:Equipment {id: eq.id})
    SET e.equipment_id = eq.id,
        e.equipment_type = eq.type,
        e.name = eq.id,
        e.vibration = COALESCE(e.vibration, eq.vibration),
        e.temp = COALESCE(e.temp, eq.temp),
        e.pressure = COALESCE(e.pressure, eq.pressure)
    """
    session.run(seed_nodes_query)

    seed_edges_query = """
    MATCH (p:Equipment {id: 'PUMP-101'}), (v:Equipment {id: 'VALVE-202'}), (t:Equipment {id: 'TURBINE-301'})
    MERGE (p)-[:CONNECTED_TO]->(v)
    MERGE (v)-[:CONNECTED_TO]->(t)
    """
    session.run(seed_edges_query)


# ۲. استخراج ساختار گراف و ویژگی‌ها از دیتابیس صنعتی Neo4j GDS
def fetch_graph_from_neo4j_gds(
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
    user: str = os.getenv("NEO4J_USER", "neo4j"), 
    password: str = os.getenv("NEO4J_PASSWORD", "password123")
) -> Tuple[dgl.DGLGraph, torch.Tensor, List[str]]:
    """
    استعلام توپولوژی خط تولید و ویژگی‌های سنسورها مستقیماً از دیتابیس Neo4j GDS
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # گام اتوماتیک: اطمینان از وجود داشتن گراف اولیه در Neo4j
            _auto_seed_neo4j_graph(session)

            # استخراج تمام گره‌های قطعات و ویژگی‌های عددی سنسورها
            nodes_query = """
            MATCH (e:Equipment)
            RETURN e.id AS id, COALESCE(e.vibration, 0.1) AS vibration, 
                   COALESCE(e.temp, 0.3) AS temp, COALESCE(e.pressure, 0.4) AS pressure
            ORDER BY e.id ASC
            """
            nodes_result = session.run(nodes_query)
            
            component_names = []
            node_id_map: Dict[str, int] = {}
            features_list = []
            
            for idx, record in enumerate(nodes_result):
                comp_id = record["id"]
                component_names.append(comp_id)
                node_id_map[comp_id] = idx
                features_list.append([record["vibration"], record["temp"], record["pressure"]])
            
            num_nodes = len(component_names)
            if num_nodes == 0:
                raise ValueError("هیچ گرهی با لیبل Equipment در Neo4j یافت نشد.")

            # استخراج اتصالات فیزیکی خط تولید
            edges_query = """
            MATCH (src:Equipment)-[:CONNECTED_TO]->(dst:Equipment)
            RETURN src.id AS src_id, dst.id AS dst_id
            """
            edges_result = session.run(edges_query)
            
            src_indices = []
            dst_indices = []
            for record in edges_result:
                if record["src_id"] in node_id_map and record["dst_id"] in node_id_map:
                    src_indices.append(node_id_map[record["src_id"]])
                    dst_indices.append(node_id_map[record["dst_id"]])

            # ساخت دوطرفه روابط (Graph Bidirectionality)
            src_tensor = torch.tensor(src_indices + dst_indices, dtype=torch.int64)
            dst_tensor = torch.tensor(dst_indices + src_indices, dtype=torch.int64)

            # مقداردهی num_nodes با تعداد واقعی گره‌ها
            g = dgl.graph((src_tensor, dst_tensor), num_nodes=num_nodes) # type: ignore
            features = torch.tensor(features_list, dtype=torch.float32)

            driver.close()
            print("✅ گراف با موفقیت در Neo4j بررسی/ساخته شد و به DGL نگاشت گردید.")
            return g, features, component_names

    except Exception as e:
        print(f"⚠️ اتصال به Neo4j برقرار نشد ({e}). استفاده از گراف پیش‌فرض محلی...")
        return create_fallback_industrial_system()


# ۳. سیستم جایگزین (Fallback) در صورت عدم دسترسی به دیتابیس
def create_fallback_industrial_system() -> Tuple[dgl.DGLGraph, torch.Tensor, List[str]]:
    src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])
    g = dgl.graph((src, dst)) # type: ignore
    
    component_names = ["PUMP-101", "VALVE-202", "TEMP-SENSOR-01", "TURBINE-301", "TANK-404"]
    features = torch.tensor([
        [0.89, 0.94, 0.82],  # PUMP-101
        [0.20, 0.45, 0.50],  # VALVE-202
        [0.15, 0.40, 0.40],  # TEMP-SENSOR-01
        [0.30, 0.60, 0.55],  # TURBINE-301
        [0.10, 0.30, 0.20]   # TANK-404
    ], dtype=torch.float32)
    
    return g, features, component_names


# ۴. تست محاسبات و آماده‌سازی Payload جهت ارسال به Graphiti
if __name__ == "__main__":
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds()
    
    # مقداردهی مدل
    model = ComponentFailureGNN(in_dim=3, hidden_dim=16)
    
    # اجرای استنتاج GNN
    node_risks = model(industrial_graph, sensor_features)
    
    print("\n✅ محاسبه ریسک قطعات توسط GNN (متصل به Neo4j) تکمیل شد:\n")
    
    graphiti_episodes: List[Dict[str, Any]] = []
    
    for idx, name in enumerate(component_names):
        risk_value = node_risks[idx].item()
        print(f"🔸 قطعه: {name:15} | احتمال خرابی: {risk_value:.2f}%")
        
        status = "Critical" if risk_value > 70.0 else "Normal"
        
        graphiti_episodes.append({
            "source": name,
            "relation": "has_status",
            "target": f"{status} (Risk: {risk_value:.1f}%)"
        })

    print("\n📦 داده ساختاریافته آماده برای ارسال به API حافظه Graphiti:")
    print(graphiti_episodes[0])