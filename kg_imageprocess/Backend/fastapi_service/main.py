from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.websocket_pose import router as websocket_router
from db.neo4j_client import neo4j_client
from db.graphiti_client import graphiti_client
from ollama_connection.ollama_client import ollama_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ۰. مقداردهی و پیکربندی اولیه Ollama
    if ollama_service.check_connection():
        ollama_service.setup_environment()
        print("🦙 سرویس Ollama محلی شناسایی شد و آماده استفاده است.")
    else:
        print("⚠️ سرویس Ollama محلی روی پورت 11434 یافت نشد! مطمئن شوید 'ollama run llama3' در حال اجراست.")

    # ۱. برقراری اتصال غیرهمگام به Neo4j
    try:
        await neo4j_client.connect()
        print("✅ اتصال غیرهمگام به Neo4j برقرار شد.")
    except Exception as e:
        print(f"⚠️ خطای اتصال به Neo4j: {e}")

    # ۲. مقداردهی اولیه موتور حافظه زمان‌مند Graphiti
    try:
        await graphiti_client.initialize()
        print("🧠 موتور Graphiti (Temporal Memory Graph) با موفقیت مقداردهی شد.")
    except Exception as e:
        print(f"⚠️ خطای مقداردهی اولیه Graphiti: {e}")

    yield

    # ۳. بستن اتصالات هنگام shutdown
    try:
        await graphiti_client.close()
    except Exception:
        pass

    try:
        await neo4j_client.close()
        print("🛑 اتصالات Neo4j و Graphiti بسته شدند.")
    except Exception:
        pass


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
    ollama_healthy = ollama_service.check_connection()
    graphiti_ready = getattr(graphiti_client, "_graphiti", None) is not None

    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "neo4j_connected": neo4j_healthy,
        "ollama_connected": ollama_healthy,
        "graphiti_initialized": graphiti_ready
    }