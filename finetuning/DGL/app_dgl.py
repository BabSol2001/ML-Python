import os
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs  # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from train_dgl import FraudDetectorDGL

model = None
g = None
features = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, g, features
    if not os.path.exists("dgl_model.pth") or not os.path.exists("dgl_graph.bin"):
        print("⚠️ فایل‌های مدل پیدا نشدند! ابتدا train_dgl.py را اجرا کنید.")
    else:
        try:
            # استفاده مستقیم از تابع load_graphs بدون پیشوند dgl
            graphs, _ = load_graphs("dgl_graph.bin")
            g = graphs[0]
            
            if "feat" in g.ndata:
                features = g.ndata["feat"]
            elif "h" in g.ndata:
                features = g.ndata["h"]
            else:
                raise KeyError("ویژگی گره‌ها در گراف یافت نشد.")

            in_feats = features.shape[1]
            model = FraudDetectorDGL(in_feats=in_feats, hidden_feats=16, num_classes=2)
            model.load_state_dict(torch.load("dgl_model.pth", weights_only=True))
            model.eval()

            print("✅ مدل و گراف DGL با موفقیت بارگذاری شدند.")
        except Exception as e:
            print(f"❌ خطا در بارگذاری فایل‌ها: {e}")
    yield

app = FastAPI(title="DGL Fraud Agent Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentReq(BaseModel):
    agent_id: int

@app.post("/detect_agent")
def detect_agent(req: AgentReq):
    if model is None or g is None or features is None:
        raise HTTPException(
            status_code=500, 
            detail="مدل بارگذاری نشده است."
        )

    if req.agent_id < 0 or req.agent_id >= g.num_nodes():
        raise HTTPException(status_code=400, detail="شناسه ایجنت معتبر نیست.")

    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits, dim=1)
        fraud_prob = probs[req.agent_id][1].item()

    return {
        "agent_id": req.agent_id,
        "fraud_risk_percent": round(fraud_prob * 100, 2),
        "is_suspicious": fraud_prob > 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)