# seed_master_exercises.py
import asyncio
from neo4j import AsyncGraphDatabase
from config import settings
from ontology.biomechanics_core_schema import MASTER_EXERCISES


async def seed_master_data():
    target_db = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")
    print(f"🚀 در حال مقداردهی اولیه نودهای مرجع روی دیتابیس [{target_db}]...")

    neo4j_uri = settings.NEO4J_URI.replace("localhost", "127.0.0.1")

    # ساخت مستقیم درایور رسمی Neo4j
    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    # ۱. ایجاد ایندکس‌ها و قوانین یکتایی
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Exercise) REQUIRE e.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (j:BodyJoint) REQUIRE j.joint_name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Athlete) REQUIRE a.user_id IS UNIQUE;"
    ]

    for constraint_cypher in constraints:
        try:
            await driver.execute_query(
                constraint_cypher,  # type: ignore
                database_=target_db
            )
        except Exception as e:
            print(f"⚠️ هشدار constraint: {e}")

    # ۲. اضافه کردن ۳ حرکت اصلی و مفاصل
    seed_cypher = """
    UNWIND $exercises AS ex
    MERGE (e:Exercise {name: ex.name})
    SET e.category = ex.category,
        e.updated_at = timestamp()
    
    WITH e, ex
    UNWIND ex.primary_joints AS joint_name
    MERGE (j:BodyJoint {joint_name: joint_name})
    MERGE (e)-[:TARGETS_JOINT]->(j)
    """

    exercises_payload = [
        {
            "name": name,
            "category": data["category"],
            "primary_joints": data["primary_joints"]
        }
        for name, data in MASTER_EXERCISES.items()
    ]

    try:
        await driver.execute_query(
            seed_cypher,  # type: ignore
            exercises=exercises_payload,
            database_=target_db
        )
        print(f"\n✅ داده‌های مرجع ۳ حرکت اصلی با موفقیت روی دیتابیس [{target_db}] ثبت شدند:")
        for name in MASTER_EXERCISES.keys():
            print(f"   📌 Exercise: {name}")
    except Exception as e:
        print(f"❌ خطا در ثبت داده‌های مرجع: {e}")
    finally:
        await driver.close()
        print("\n🛑 اتصال Neo4j بسته شد.")


if __name__ == "__main__":
    asyncio.run(seed_master_data())