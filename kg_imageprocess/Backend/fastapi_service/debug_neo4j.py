import asyncio
from neo4j import AsyncGraphDatabase
from config import settings

async def debug():
    uri = settings.NEO4J_URI.replace("localhost", "127.0.0.1")
    auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    db_name = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")

    user_id = "c6fdd048-31c8-41e5-a662-3e24cf07ecf6"

    async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
        # ۱. بررسی وجود گره User
        q1 = "MATCH (u:User {id: $user_id}) RETURN u"
        records1, _, _ = await driver.execute_query(q1, user_id=user_id, database_=db_name)
        print(f"--- 1. User Node Exists? {len(records1) > 0}")
        if records1:
            print("User Data:", dict(records1[0]["u"]))

        # ۲. بررسی تمام روابط متصل به این User
        q2 = """
        MATCH (u:User {id: $user_id})-[r]-(n) 
        RETURN type(r) AS rel, labels(n) AS labels, properties(n) AS props
        """
        records2, _, _ = await driver.execute_query(q2, user_id=user_id, database_=db_name)
        print(f"\n--- 2. Connected Nodes Count: {len(records2)}")
        for record in records2:
            print(f"Rel: {record['rel']} -> Node Labels: {record['labels']}")
            print(f"     Props: {record['props']}\n")

asyncio.run(debug())