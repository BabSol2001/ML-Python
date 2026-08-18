import os
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# فراخوانی معماری مدل از فایل فاین‌تیونینگ
from train_pyg import ToolPredictorPyG

# ۱. مقداردهی اولیه متغیرهای سراسری
model = None
graph_data = None

# ۲. مدیریت بارگذاری مدل در زمان استارت FastAPI (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, graph_data
    if not os.path.exists("pyg_tool_model.pth") or not os.path.exists("pyg_data.pt"):
        print("⚠️ فایل‌های مدل پیدا نشدند! ابتدا train_pyg.py را اجرا کنید.")
    else:
        try:
            model = ToolPredictorPyG(in_channels=8, hidden_channels=16, out_channels=8)
            model.load_state_dict(torch.load("pyg_tool_model.pth", weights_only=True))
            model.eval()
            graph_data = torch.load("pyg_data.pt", weights_only=False)
            print("✅ مدل و داده‌های PyG با موفقیت در API بارگذاری شدند.")
        except Exception as e:
            print(f"❌ خطا در بارگذاری فایل‌ها: {e}")
    yield

app = FastAPI(title="PyG Tool Recommendation API", lifespan=lifespan)

# ۳. تنظیمات CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkReq(BaseModel):
    task_id: int
    tool_id: int

@app.post("/predict_tool")
def predict_tool(req: LinkReq):
    if model is None or graph_data is None:
        raise HTTPException(
            status_code=500, 
            detail="مدل بارگذاری نشده است. ابتدا train_pyg.py را اجرا کنید."
        )

    if req.task_id > 4 or req.tool_id > 4 or req.task_id < 0 or req.tool_id < 0:
        raise HTTPException(
            status_code=400, 
            detail="شناسه گره‌ها باید بین 0 تا 4 باشد."
        )
    
    with torch.no_grad():
        z = model.encoder(graph_data.x, graph_data.edge_index)
        edge = torch.tensor([[req.task_id], [req.tool_id]], dtype=torch.long)
        score = torch.sigmoid(model.decode(z, edge)).item()

    return {
        "task_id": req.task_id,
        "tool_id": req.tool_id,
        "match_score": round(score * 100, 2),
        "recommended": score > 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)