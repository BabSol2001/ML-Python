import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

memory_data = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_data
    if not os.path.exists("graphiti_memory.json"):
        print("⚠️ فایل حافظه Graphiti یافت نشد. ابتدا pipeline_graphiti.py را اجرا کنید.")
    else:
        with open("graphiti_memory.json", "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        print("✅ حافظه زمان‌مند Graphiti بارگذاری شد.")
    yield

app = FastAPI(title="Graphiti Temporal Agent Memory API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryMemoryReq(BaseModel):
    subject: str

@app.post("/get_active_memory")
def get_active_memory(req: QueryMemoryReq):
    if not memory_data:
        raise HTTPException(status_code=500, detail="حافظه بارگذاری نشده است.")

    # استخراج فقط فکت‌های معتبر که زمان انقضا ندارند (invalidated_at is None)
    active_facts = [
        f"{item['source']} {item['relation']} {item['target']}"
        for item in memory_data
        if item['source'].lower() == req.subject.lower() and item['invalidated_at'] is None
    ]

    return {
        "subject": req.subject,
        "active_facts": active_facts,
        "total_episodes_in_history": len(memory_data)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)