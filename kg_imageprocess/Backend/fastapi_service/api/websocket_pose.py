import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from schemas.pose_schema import PoseFramePayload, PoseFeedbackResponse
from core.rule_engine import check_squat_rules
from services.django_client import fetch_athlete_context, check_user_subscription_status

logger = logging.getLogger("WebSocketPose")
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


@router.websocket("/ws/pose/{user_id}")
async def websocket_pose_endpoint(websocket: WebSocket, user_id: str):
    # ۱. پذیرش اولیه اتصال جهت بررسی‌های اعتبارسنجی
    await manager.connect(websocket)
    logger.info(f"🔌 درخواست اتصال وب‌سوکت برای کاربر: {user_id}")

    # ۲. استعلام وضعیت اشتراک و سقف استفاده از اپلیکیشن payments جانگو
    sub_check = await check_user_subscription_status(user_id)
    if not sub_check["is_allowed"]:
        error_payload = {
            "error": "SUBSCRIPTION_EXPIRED",
            "message": sub_check["reason"],
            "remaining_analyses": sub_check["remaining_analyses"]
        }
        await websocket.send_text(json.dumps(error_payload))
        manager.disconnect(websocket)
        await websocket.close(code=4003)
        logger.warning(f"⛔ عدم اجازه اتصال به کاربر {user_id}: {sub_check['reason']}")
        return

    # ۳. دریافت context جامع بیومکانیکی و سوابق پزشکی کاربر از جانگو
    athlete_context = await fetch_athlete_context(user_id)
    logger.info(f"✅ پرونده کاربر {user_id} دریافت شد. اعمال آستانه‌های پویا...")

    try:
        while True:
            # دریافت پیام خام متنی از فلاتر
            data = await websocket.receive_text()

            try:
                # اعتبارسنجی ساختار JSON ورودی
                json_data = json.loads(data)
                payload = PoseFramePayload(**json_data)

                # تحلیل لایو حرکت با Rule Engine + تزریق Athlete Context
                feedback: PoseFeedbackResponse = check_squat_rules(
                    payload=payload, 
                    athlete_context=athlete_context
                )

                # ارسال بازخورد آنی زیر چند میلی‌ثانیه به کاربر
                await websocket.send_text(feedback.model_dump_json())

            except ValidationError as e:
                error_response = {
                    "error": "Invalid payload format",
                    "details": e.errors()
                }
                await websocket.send_text(json.dumps(error_response))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"❌ اتصال وب‌سوکت کاربر {user_id} قطع شد.")
    except Exception as e:
        logger.error(f"خطای پیش‌بینی نشده در وب‌سوکت کاربر {user_id}: {e}")
        manager.disconnect(websocket)
        await websocket.close()