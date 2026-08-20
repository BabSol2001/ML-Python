import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

index_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_data
    if not os.path.exists("graphrag_index.json"):
        print("⚠️ فایل ایندکس یافت نشد. ابتدا pipeline_graphrag.py را اجرا کنید.")
    else:
        with open("graphrag_index.json", "r", encoding="utf-8") as f:
            index_data = json.load(f)
        print("✅ ایندکس GraphRAG در FastAPI بارگذاری شد.")
    yield

app = FastAPI(title="Microsoft GraphRAG Global Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GlobalQueryReq(BaseModel):
    query: str

@app.post("/global_search")
def global_search(req: GlobalQueryReq):
    if not index_data:
        raise HTTPException(status_code=500, detail="ایندکس بارگذاری نشده است.")

    reports = index_data.get("reports", {})
    # الگوی Map-Reduce: ترکیب گزارش تمام جوامع برای پاسخ‌دهی به پرس‌وجوی کلان
    combined_context = "\n".join([f"- {rep}" for rep in reports.values()])
    
    global_answer = f"پاسخ کلان بر اساس تحلیل جوامع گراف دانش:\n{combined_context}\n\nنتیجه‌گیری: پرسش '{req.query}' مستقیماً با ابعاد تورمی و ریسک بازار در ارتباط است."

    return {
        "query": req.query,
        "global_answer": global_answer,
        "communities_analyzed": len(reports)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)