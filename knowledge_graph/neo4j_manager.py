from typing import List, Tuple
from neo4j import GraphDatabase

# اتصال به فایل اصلی
from networkx_triplets import KnowledgeGraph, build_sample_kg, Triplet

class Neo4jKnowledgeGraphManager:
    def __init__(self, uri: str = "bolt://localhost:7687", auth: Tuple[str, str] = ("neo4j", "nastaran1614")):
        """مدیریت اتصال به پایگاه داده Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def sync_kg_to_neo4j(self, kg: KnowledgeGraph):
        """انتقال تمام سه‌تایی‌های موجود در KnowledgeGraph پایتون به Neo4j"""
        cypher_query = """
        UNWIND $batch AS row
        MERGE (h:Entity {name: row.head})
        MERGE (t:Entity {name: row.tail})
        MERGE (h)-[r:CONNECTED_TO {type: row.relation}]->(t)
        """
        
        # تبدیل سه‌تایی‌ها به لیست دیکشنری‌های استاندارد
        batch_data = [{"head": t.head, "relation": t.relation, "tail": t.tail} for t in kg.triplets]

        with self.driver.session() as session:
            # ارسال مستقیم پارامتر batch به Cypher
            session.run(cypher_query, batch=batch_data)
            print(f"✅ تعداد {len(kg.triplets)} سه‌تایی با موفقیت در دیتابیس Neo4j همگام‌سازی شد.")

    def cypher_impact_analysis(self, source_entity: str) -> List[str]:
        """اجرای مستقیم تحلیل اثر بحران در Neo4j"""
        cypher_query = """
        MATCH (source:Entity {name: $source_name})-[*1..5]->(affected:Entity)
        RETURN DISTINCT affected.name AS affected_entity
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, source_name=source_entity)
            return [record["affected_entity"] for record in result]

# ==========================================
# تست مستقل ماژول Neo4j
# ==========================================
if __name__ == "__main__":
    kg = build_sample_kg()
    
    print("🔌 در حال اتصال به دیتابیس Neo4j...")
    try:
        neo4j_db = Neo4jKnowledgeGraphManager(
            uri="bolt://localhost:7687", 
            auth=("neo4j", "nastaran1614")
        )
        
        # ۱. همگام‌سازی داده‌ها
        neo4j_db.sync_kg_to_neo4j(kg)
        
        # ۲. اجرا و تست کوئری در Neo4j
        risk_node = "Supplier_Alpha"
        affected = neo4j_db.cypher_impact_analysis(risk_node)
        
        print(f"\n🔍 [Neo4j Cypher] گره‌های متأثر از '{risk_node}':")
        for entity in affected:
            print(f"  • {entity}")
            
        neo4j_db.close()
    except Exception as e:
        print(f"\n⚠️ خطا در اجرای عملیات Neo4j:")
        print(f"جزئیات: {e}")