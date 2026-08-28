from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.websocket_pose import router as websocket_router
from db.neo4j_client import neo4j_client
from db.graphiti_client import graphiti_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ۱. برقراری اتصال غیرهمگام به Neo4j
    await neo4j_client.connect()
    print("✅ اتصال غیرهمگام به Neo4j برقرار شد.")
    
    # ۲. مقداردهی اولیه موتور حافظه زمان‌مند Graphiti
    await graphiti_client.initialize()
    print("🧠 موتور Graphiti (Temporal Memory Graph) با موفقیت مقداردهی شد.")
    
    yield
    
    # ۳. بستن اتصالات هنگام shutdown
    await graphiti_client.close()
    await neo4j_client.close()
    print("🛑 اتصالات Neo4j و Graphiti بسته شدند.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    debug=settings.DEBUG,
    version="1.0.0",
    description="سرویس پردازش آنی حرکات ورزشی و استدلال گرافی",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router, tags=["Live Pose Processing"])


@app.get("/health", tags=["System"])
async def health_check():
    neo4j_healthy = await neo4j_client.check_connection()
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "neo4j_connected": neo4j_healthy,
        "graphiti_initialized": graphiti_client._graphiti is not None
    }