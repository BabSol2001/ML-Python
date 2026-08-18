import os
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# فراخوانی معماری مدل از فایل آموزش
from train_gnn import LinkPredictor

# ۱. تعریف دقیق مسیر فایل‌ها نسبت به پوشه جاری
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gnn_model.pth")
DATA_PATH = os.path.join(BASE_DIR, "graph_data.pt")

# ۲. مقداردهی اولیه متغیرهای سراسری
graph_data = None
tuned_model = None

# ۳. مدیریت بارگذاری مدل با استفاده از Lifespan (جایگزین on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_data, tuned_model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("⚠️ فایل‌های مدل پیدا نشدند! ابتدا اسکریپت train_gnn.py را اجرا کنید.")
    else:
        try:
            graph_data = torch.load(DATA_PATH, weights_only=False)
            tuned_model = LinkPredictor(in_channels=16, hidden_channels=32, out_channels=16)
            tuned_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            tuned_model.eval()
            print("✅ مدل و داده‌های گراف با موفقیت بارگذاری شدند.")
        except Exception as e:
            print(f"❌ خطا در بارگذاری فایل‌های مدل: {e}")
    yield

app = FastAPI(title="GNN Graph Tuning API", lifespan=lifespan)

# ۴. حل مشکل CORS و هندل کردن درخواست‌های OPTIONS مرورگر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    source_node: int
    target_node: int

@app.post("/predict_link")
def predict_link(req: PredictionRequest):
    # بررسی لود بودن مدل پیش از پردازش درخواست
    if graph_data is None or tuned_model is None:
        raise HTTPException(
            status_code=500, 
            detail="مدل بارگذاری نشده است. ابتدا train_gnn.py را اجرا کرده و سرور را ریستارت کنید."
        )

    if graph_data.x is None:
        raise HTTPException(status_code=500, detail="ویژگی‌های گره یافت نشد.")
    
    # اعتبارسنجی دامنه ID گره‌ها
    num_nodes = graph_data.x.size(0)
    if req.source_node >= num_nodes or req.target_node >= num_nodes or req.source_node < 0 or req.target_node < 0:
        raise HTTPException(
            status_code=400, 
            detail=f"شناسه گره باید بین 0 تا {num_nodes - 1} باشد."
        )

    with torch.no_grad():
        z = tuned_model.encoder(graph_data.x, graph_data.edge_index)
        edge = torch.tensor([[req.source_node], [req.target_node]], dtype=torch.long)
        score = torch.sigmoid(tuned_model.decode(z, edge)).item()
        
    return {
        "source_node": req.source_node,
        "target_node": req.target_node,
        "link_probability": round(score, 4),
        "exists": score > 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)