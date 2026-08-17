from typing import List, Tuple, Dict, Any
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
        
        batch_data = [{"head": t.head, "relation": t.relation, "tail": t.tail} for t in kg.triplets]

        with self.driver.session() as session:
            session.run(cypher_query, batch=batch_data)
            print(f"✅ تعداد {len(kg.triplets)} سه‌تایی با موفقیت در دیتابیس Neo4j همگام‌سازی شد.")

    def cypher_impact_analysis(self, source_entity: str) -> List[str]:
        """تحلیل اثر بحران/روابط در Neo4j بدون محدودیت Label و غیرحساس به حروف کوچک و بزرگ"""
        cypher_query = """
        MATCH (source)
        WHERE toLower(source.name) = toLower($source_name)
        MATCH (source)-[*1..2]-(affected)
        WHERE source <> affected AND affected.name IS NOT NULL
        RETURN DISTINCT affected.name AS affected_entity
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, source_name=source_entity)
            return [record["affected_entity"] for record in result if record["affected_entity"]]

def get_entity_context(self, entity_name: str) -> List[Dict[str, Any]]:
        """استخراج جامع روابط گراف با جستجوی عام روی تمامی ویژگی‌ها"""
        clean_name = entity_name.strip().lower() if entity_name else ""
        
        # ۱. کوئری جامع برای پیدا کردن هر گرهی که شاملی کلمه درخواستی است
        cypher_query = """
        MATCH (source)-[r]-(target)
        WHERE any(k IN keys(source) WHERE toLower(toString(source[k])) CONTAINS $entity_name)
           OR any(lbl IN labels(source) WHERE toLower(lbl) CONTAINS $entity_name)
        RETURN 
            coalesce(source.name, source.id, source.title, labels(source)[0], 'Node') AS head,
            coalesce(r.type, type(r), 'RELATED_TO') AS relation,
            coalesce(target.name, target.id, target.title, labels(target)[0], 'Node') AS tail
        LIMIT 40
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, entity_name=clean_name)
            triplets = []
            for record in result:
                triplets.append({
                    "head": str(record["head"]),
                    "relation": str(record["relation"]),
                    "tail": str(record["tail"])
                })
            
            # ۲. مکانیزم پشتیبان: اگر کلید دقیقی یافت نشد، ۳۰ رابطه کلی گراف ارسال می‌شود
            if not triplets:
                fallback_query = """
                MATCH (source)-[r]->(target)
                RETURN 
                    coalesce(source.name, source.id, source.title, labels(source)[0], 'Node') AS head,
                    coalesce(r.type, type(r), 'RELATED_TO') AS relation,
                    coalesce(target.name, target.id, target.title, labels(target)[0], 'Node') AS tail
                LIMIT 30
                """
                res_fb = session.run(fallback_query)
                for record in res_fb:
                    triplets.append({
                        "head": str(record["head"]),
                        "relation": str(record["relation"]),
                        "tail": str(record["tail"])
                    })
                    
            return triplets


# ==========================================
# تست مستقل ماژول Neo4j
# ==========================================
if __name__ == "__main__":
    print("🔌 در حال اتصال به دیتابیس Neo4j...")
    try:
        neo4j_db = Neo4jKnowledgeGraphManager(
            uri="bolt://localhost:7687", 
            auth=("neo4j", "nastaran1614")
        )
        
        # تست با گره واقعاً موجود در دیتابیس شما
        test_node = "Neo4j"
        affected = neo4j_db.cypher_impact_analysis(test_node)
        
        print(f"\n🔍 [Neo4j Cypher] گره‌های متأثر/مرتبط با '{test_node}':")
        for entity in affected:
            print(f"  • {entity}")
            
        neo4j_db.close()
    except Exception as e:
        print(f"\n⚠️ خطا در اجرای عملیات Neo4j:")
        print(f"جزئیات: {e}")