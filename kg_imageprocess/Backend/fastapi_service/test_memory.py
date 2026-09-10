import asyncio
from db.graphiti_client import graphiti_client

async def test():
    await graphiti_client.initialize()
    res = await graphiti_client.search_user_memory(
        user_id="c6fdd048-31c8-41e5-a662-3e24cf07ecf6", 
        query="خطای تمرین"
    )
    print("RESULT:", res)
    await graphiti_client.close()

asyncio.run(test())