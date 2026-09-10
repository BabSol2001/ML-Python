# test_step2.py
import asyncio
from db.neo4j_client import neo4j_client
from db.graphiti_client import graphiti_client

async def run_test():
    print("--- ۱. تست اتصال Neo4j ---")
    await neo4j_client.connect()
    is_neo4j_ok = await neo4j_client.check_connection()
    print(f"Neo4j Status: {'✅ Connected' if is_neo4j_ok else '❌ Failed'}")

    if is_neo4j_ok:
        session_id = await neo4j_client.create_session_node("test_user_123", "squat")
        print(f"Session Created in Neo4j with ID: {session_id}")
        
        await neo4j_client.log_frame_analysis(session_id, frame_id=1, knee_angle=105.5, is_valid=False, error_code="NOT_DEEP_ENOUGH")
        print("Frame logged to Neo4j successfully.")

    print("\n--- ۲. تست اتصال و ثبت فکت در Graphiti ---")
    try:
        await graphiti_client.initialize()
        await graphiti_client.log_biomechanical_fact(
            user_id="test_user_123",
            session_id="session_test_99",
            error_code="NOT_DEEP_ENOUGH",
            details="زاویه زانو ۱۰۵ درجه بود که از حد مجاز ۹۰ درجه بیشتر است."
        )
        print("Fact logged to Graphiti successfully.")
        
        memories = await graphiti_client.search_user_memory("test_user_123", "knee angle errors")
        print(f"Retrieved Facts from Graphiti: {len(memories)}")
        for m in memories:
            print(f" - {m['fact']}")
    except Exception as e:
        print(f"Graphiti Test Warning: {e}")

    await neo4j_client.close()
    await graphiti_client.close()

if __name__ == "__main__":
    asyncio.run(run_test())