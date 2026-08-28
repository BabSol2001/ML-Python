"""
========================================================================================
کد دقیقاً چکار می‌کند؟
========================================================================================
این اسکریپت لایه محاسباتی GNN را با ترکیب ۳ الگوریتم قدرتمند Neo4j GDS پیاده‌سازی می‌کند:
۱. ساخت اتوماتیک ۱۰۰ تجهیز صنعتی و اتصالات در دیتابیس Neo4j هنگام استارت‌آپ.
۲. ساخت In-Memory Graph Projection در رم دیتابیس Neo4j GDS.
۳. اجرای همزمان FastRP (بردار ساختاری)، PageRank (اهمیت قطعه) و Louvain (خوشه‌بندی زون‌ها).
۴. استخراج یکپارچه خصوصیات با `gds.nodeProperty.stream`.
۵. تغذیه ماتریس ویژگی ۷ بعدی (۳ سنسور + ۴ ویژگی GDS) به شبکه عصبی DGL GNN.
۶. اضافه کردن Self-Loop اتوماتیک جهت رفع خطای 0-in-degree در DGL.
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


# ۱. تعریف ساختار شبکه عصبی گرافی (DGL GNN Model با ورودی ۷ بعدی)
class ComponentFailureGNN(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 16, out_dim: int = 1):
        super(ComponentFailureGNN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)  # type: ignore
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)  # type: ignore
        self.node_regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        component_risks = torch.sigmoid(self.node_regressor(h)) * 100.0
        return component_risks


# ۲. ایجاد اتوماتیک ۱۰۰ نود صنعتی و اتصالات در Neo4j (تراکنش مستقل جهت Commit قطعی)
def _auto_seed_100_nodes(driver):
    """ساخت اتوماتیک ۱۰۰ نود با استفاده از Write Transaction صریح برای تضمین Commit در Neo4j 5"""
    seed_nodes_query = """
    UNWIND range(1, 100) AS id_num
    WITH id_num, 
         CASE 
           WHEN id_num % 4 = 0 THEN 'PUMP-' + toString(id_num)
           WHEN id_num % 4 = 1 THEN 'VALVE-' + toString(id_num)
           WHEN id_num % 4 = 2 THEN 'MOTOR-' + toString(id_num)
           ELSE 'SENSOR-' + toString(id_num)
         END AS eq_id,
         CASE 
           WHEN id_num % 4 = 0 THEN 'Centrifugal Pump'
           WHEN id_num % 4 = 1 THEN 'Solenoid Valve'
           WHEN id_num % 4 = 2 THEN 'Electric Motor'
           ELSE 'Vibration Sensor'
         END AS eq_type
    MERGE (e:Equipment {id: eq_id})
    SET e.equipment_id = eq_id,
        e.equipment_type = eq_type,
        e.name = eq_id,
        e.vibration = COALESCE(e.vibration, round(rand() * 0.8 + 0.1, 2)),
        e.temp = COALESCE(e.temp, round(rand() * 0.7 + 0.2, 2)),
        e.pressure = COALESCE(e.pressure, round(rand() * 0.6 + 0.3, 2))
    """
    
    seed_edges_query = """
    MATCH (e:Equipment)
    WITH e ORDER BY e.id
    WITH collect(e) AS eq_list
    UNWIND range(0, size(eq_list)-2) AS i
    WITH eq_list[i] AS src, eq_list[i+1] AS dst
    MERGE (src)-[:CONNECTED_TO]->(dst)
    """

    with driver.session(database="neo4j") as session:
        session.execute_write(lambda tx: tx.run(seed_nodes_query))
        session.execute_write(lambda tx: tx.run(seed_edges_query))
    print("✅ ۱۰۰ نود و اتصالات با موفقیت در Neo4j Commit شدند.")

# ۳. استخراج چندگانه از GDS (FastRP + PageRank + Louvain)
def fetch_graph_from_neo4j_gds(
    uri: str = os.getenv("NEO4J_URI", "bolt://neo4j-gds:7687"), 
    user: str = os.getenv("NEO4J_USER", "neo4j"), 
    password: str = os.getenv("NEO4J_PASSWORD", "password123")
) -> Tuple[dgl.DGLGraph, torch.Tensor, List[str]]:
    """
    اجرای FastRP، PageRank و Louvain درون GDS و ساخت ماتریس ویژگی 7 بعدی
    """
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # ۱. ابتدا ساخت ۱۰۰ نود را در دیتابیس Commit کامل می‌کنیم
        _auto_seed_100_nodes(driver)

        # ۲. پردازش‌های GDS در Session جدید
        with driver.session() as session:
            # پاکسازی پروژکشن قدیمی در صورت وجود
            try:
                session.run("CALL gds.graph.drop('industrialPipeline', false) YIELD graphName")
            except Exception:
                pass

            # گام ۱: ساخت In-Memory Graph Projection
            gds_project_query = """
            CALL gds.graph.project(
              'industrialPipeline',
              'Equipment',
              'CONNECTED_TO',
              { nodeProperties: ['vibration', 'temp', 'pressure'] }
            )
            """
            session.run(gds_project_query)

            # گام ۲: اجرای FastRP با mutate
            session.run("""
            CALL gds.fastRP.mutate('industrialPipeline', {
              mutateProperty: 'fastrp_emb',
              embeddingDimension: 2,
              nodeSelfInfluence: 0.5
            })
            """)

            # گام ۳: اجرای PageRank با mutate
            session.run("""
            CALL gds.pageRank.mutate('industrialPipeline', {
              mutateProperty: 'pagerank_score',
              maxIterations: 20,
              dampingFactor: 0.85
            })
            """)

            # گام ۴: اجرای Louvain با mutate
            session.run("""
            CALL gds.louvain.mutate('industrialPipeline', {
              mutateProperty: 'community_id'
            })
            """)

            # گام ۵: خواندن یکپارچه تمامی ویژگی‌ها از پروژکشن
            stream_query = """
            CALL gds.graph.nodeProperties.stream(
              'industrialPipeline',
              ['fastrp_emb', 'pagerank_score', 'community_id']
            )
            YIELD nodeId, nodeProperty, propertyValue
            RETURN gds.util.asNode(nodeId).id AS id,
                   gds.util.asNode(nodeId).vibration AS vibration,
                   gds.util.asNode(nodeId).temp AS temp,
                   gds.util.asNode(nodeId).pressure AS pressure,
                   nodeProperty,
                   propertyValue
            ORDER BY id ASC
            """
            stream_result = session.run(stream_query)

            # سازماندهی داده‌های دریافتی به ازای هر نود
            nodes_data: Dict[str, Dict[str, Any]] = {}
            for record in stream_result:
                node_id = record["id"]
                if node_id not in nodes_data:
                    nodes_data[node_id] = {
                        "vibration": float(record["vibration"] or 0.5),
                        "temp": float(record["temp"] or 0.5),
                        "pressure": float(record["pressure"] or 0.5),
                        "fastrp": [0.0, 0.0],
                        "pagerank": 0.0,
                        "community": 0.0
                    }

                prop_name = record["nodeProperty"]
                prop_val = record["propertyValue"]

                if prop_name == "fastrp_emb" and isinstance(prop_val, list):
                    nodes_data[node_id]["fastrp"] = [float(x) for x in prop_val[:2]]
                elif prop_name == "pagerank_score":
                    nodes_data[node_id]["pagerank"] = float(prop_val)
                elif prop_name == "community_id":
                    nodes_data[node_id]["community"] = float(prop_val)

            # مرتب‌سازی کل نودها و ساخت ماتریس ویژگی ۷ بعدی
            component_names = sorted(list(nodes_data.keys()))
            node_id_map: Dict[str, int] = {comp_id: i for i, comp_id in enumerate(component_names)}
            features_list = []

            for comp_id in component_names:
                d = nodes_data[comp_id]
                features_list.append([
                    d["vibration"], 
                    d["temp"], 
                    d["pressure"], 
                    d["fastrp"][0], 
                    d["fastrp"][1], 
                    d["pagerank"], 
                    d["community"]
                ])

            # استخراج یال‌ها برای DGL
            edges_query = "MATCH (src:Equipment)-[:CONNECTED_TO]->(dst:Equipment) RETURN src.id AS src_id, dst.id AS dst_id"
            edges_result = session.run(edges_query)

            src_indices, dst_indices = [], []
            for record in edges_result:
                if record["src_id"] in node_id_map and record["dst_id"] in node_id_map:
                    src_indices.append(node_id_map[record["src_id"]])
                    dst_indices.append(node_id_map[record["dst_id"]])

            # پاکسازی پروژکشن از RAM
            try:
                session.run("CALL gds.graph.drop('industrialPipeline', false) YIELD graphName")
            except Exception:
                pass

            num_nodes = len(component_names)
            src_tensor = torch.tensor(src_indices + dst_indices, dtype=torch.int64)
            dst_tensor = torch.tensor(dst_indices + src_indices, dtype=torch.int64)

            # ساخت گراف DGL
            g = dgl.graph((src_tensor, dst_tensor), num_nodes=num_nodes)  # type: ignore
            
            # 🛠️ اضافه کردن Self-Loop جهت جلوگیری از خطای 0-in-degree در DGL
            try:
                g = dgl.add_self_loop(g)  # type: ignore
            except Exception:
                pass

            features = torch.tensor(features_list, dtype=torch.float32)

            driver.close()
            print(f"✅ تعداد {num_nodes} نود با FastRP + PageRank + Louvain از Neo4j GDS پردازش شد (ویژگی ۷ بعدی).")
            return g, features, component_names

    except Exception as e:
        if driver:
            try:
                driver.close()
            except Exception:
                pass
        print(f"⚠️ خطای اتصال یا عدم وجود GDS ({e}). اجرای Fallback...")
        return create_fallback_industrial_system()


# ۴. سیستم جایگزین (Fallback)
def create_fallback_industrial_system() -> Tuple[dgl.DGLGraph, torch.Tensor, List[str]]:
    component_names = [f"EQUIPMENT-{i:03d}" for i in range(1, 101)]
    src = torch.tensor([i for i in range(99)])
    dst = torch.tensor([i+1 for i in range(99)])
    g = dgl.graph((src, dst), num_nodes=100)  # type: ignore
    
    try:
        g = dgl.add_self_loop(g)  # type: ignore
    except Exception:
        pass

    features = torch.rand((100, 7), dtype=torch.float32)
    return g, features, component_names


# ۵. اجرای اتوماتیک هنگام فراخوانی فایل در FastAPI Startup
def init_database_on_startup():
    fetch_graph_from_neo4j_gds()


if __name__ == "__main__":
    industrial_graph, sensor_features, component_names = fetch_graph_from_neo4j_gds()
    model = ComponentFailureGNN(in_dim=7, hidden_dim=16)
    node_risks = model(industrial_graph, sensor_features)
    
    print(f"\n✅ محاسبه ریسک ۷ بعدی GNN انجام شد:")
    print(f"🔹 نمونه ریسک قطعه اول ({component_names[0]}): {node_risks[0].item():.2f}%")
    print(f"🔹 نمونه ریسک قطعه آخر ({component_names[-1]}): {node_risks[-1].item():.2f}%")