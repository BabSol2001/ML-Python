import os
import shutil
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ماژول‌های پروژه
from networkx_triplets import build_sample_kg
from llm_agent import LLMGraphRAGAgent
from neo4j_manager import Neo4jKnowledgeGraphManager
from pdf_processor import PDFGraphExtractor

# بارگذاری متغیرهای محیطی از .env
load_dotenv()

app = FastAPI(title="SmartBiz-KG API", version="2.0")

# مجوز دادن به درخواست‌های Flutter (وب، دسکتاپ و موبایل)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# آماده‌سازی اولیه گراف دانش و ایجنت
kg = build_sample_kg()
llm_agent = LLMGraphRAGAgent(kg=kg)


# ==========================================
# مدل‌های داده (Data Schemas)
# ==========================================
class QueryRequest(BaseModel):
    question: str
    target_entity: Optional[str] = "Supplier_Alpha"

class ImpactRequest(BaseModel):
    source_entity: str


# ==========================================
# ۱. امکانات قبلی (پشتیبانی کامل و بدون تغییر)
# ==========================================

@app.get("/")
def read_root():
    return {"status": "online", "message": "SmartBiz-KG API is running securely"}

@app.post("/ask")
def ask_graph_agent(req: QueryRequest):
    """پاسخ‌گویی ایجنت Graph-RAG به سوال کاربر با Groq"""
    try:
        entity = req.target_entity or "Supplier_Alpha"
        answer = llm_agent.answer_question(req.question, target_entity=entity)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impact-analysis")
def get_impact_analysis(req: ImpactRequest):
    """دریافت مستقیم لیست گره‌های متأثر از Neo4j"""
    try:
        neo4j_db = Neo4jKnowledgeGraphManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
            auth=(
                os.getenv("NEO4J_USER", "neo4j"), 
                os.getenv("NEO4J_PASSWORD", "nastaran1614")
            )
        )
        affected = neo4j_db.cypher_impact_analysis(req.source_entity)
        neo4j_db.close()
        return {"source_entity": req.source_entity, "affected_entities": affected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ۲. امکان جدید (آپلود PDF و ساخت اتوماتیک گراف)
# ==========================================

@app.post("/upload-pdf")
async def upload_pdf_file(file: UploadFile = File(...)):
    """دریافت فایل PDF، استخراج سه‌تایی‌ها با Groq و تزریق مستقیم به Neo4j"""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فقط فایل‌های با پسوند PDF پشتیبانی می‌شوند.")
    
    # بررسی وجود API Key جهت جلوگیری از خطای تایپ و Runtime
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500, 
            detail="کلید GROQ_API_KEY در فایل .env تعریف نشده است."
        )

    # ذخیره موقت فایل آپلودشده جهت پردازش
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # اتصال به Neo4j
        neo4j_db = Neo4jKnowledgeGraphManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
            auth=(
                os.getenv("NEO4J_USER", "neo4j"), 
                os.getenv("NEO4J_PASSWORD", "nastaran1614")
            )
        )
        
        # فراخوانی ماژول پردازش PDF
        extractor = PDFGraphExtractor(
            groq_api_key=groq_api_key,
            neo4j_manager=neo4j_db
        )
        
        # استخراج دانش و ذخیره در دیتابیس
        extracted_triplets = extractor.process_pdf_file(temp_path)
        neo4j_db.close()
        
        return {
            "status": "success",
            "message": "فایل PDF با موفقیت تحلیل و به گراف دانش منتقل شد.",
            "filename": file.filename,
            "extracted_triplets_count": len(extracted_triplets),
            "triplets": extracted_triplets
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش فایل: {str(e)}")
    finally:
        # پاک‌سازی فایل موقت از روی دیسک
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# اجرای سرور
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)