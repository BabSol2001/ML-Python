import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from schemas.pose_schema import PoseFramePayload, PoseFeedbackResponse
from core.rule_engine import check_squat_rules # ماژول تحلیل که در گام بعد می‌سازیم

router = APIRouter()

class ConnectionManager:
    """مدیریت اتصالات لایو وب‌سوکت"""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@router.websocket("/ws/pose")
async def websocket_pose_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # ۱. دریافت پیام خام متنی از فلاتر
            data = await websocket.receive_text()
            
            try:
                # ۲. اعتبار سنجی ساختار JSON با Pydantic
                json_data = json.loads(data)
                payload = PoseFramePayload(**json_data)
                
                # ۳. تحلیل حرکت با Rule Engine (محاسبه لایو)
                feedback: PoseFeedbackResponse = check_squat_rules(payload)

                # ۴. ارسال بازخورد آنی زیر چند میلی‌ثانیه به فلاتر
                await websocket.send_text(feedback.model_dump_json())

            except ValidationError as e:
                # در صورت اشتباه بودن فرمت داده ورودی
                error_response = {
                    "error": "Invalid payload format",
                    "details": e.errors()
                }
                await websocket.send_text(json.dumps(error_response))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)