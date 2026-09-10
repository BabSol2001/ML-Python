import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from schemas.pose_schema import PoseFramePayload, PoseFeedbackResponse
from core.rule_engine import check_squat_rules
from core.pose_calculator import calculate_angle_2d
from services.django_client import fetch_athlete_context, check_user_subscription_status
from ..db.neo4j_client import neo4j_client
from ..db.graphiti_client import graphiti_client

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
    logger.info(f"✅ پرونده کاربر {user_id} دریافت شد.")

    # ۴. ایجاد جلسه تمرین جدید (WorkoutSession) در Neo4j
    session_id = str(uuid.uuid4())
    try:
        if neo4j_client._driver:
            await neo4j_client.create_session_node(
                user_id=user_id, 
                exercise_name="squat", 
                session_uuid=session_id
            )
    except Exception as e:
        logger.error(f"خطا در ایجاد گره جلسه تمرین در Neo4j: {e}")

    try:
        while True:
            # دریافت پیام خام متنی از فلاتر
            data = await websocket.receive_text()

            try:
                # اعتبارسنجی ساختار JSON ورودی
                json_data = json.loads(data)
                payload = PoseFramePayload(**json_data)

                # ۵. تحلیل لایو حرکت با Rule Engine + تزریق Athlete Context
                feedback: PoseFeedbackResponse = check_squat_rules(
                    payload=payload, 
                    athlete_context=athlete_context
                )

                # ۶. استخراج زاویه واقعی زانو برای ثبت در گراف
                keypoints_dict = {kp.id: (kp.x, kp.y) for kp in payload.keypoints if kp.score > 0.5}
                knee_angle = 0.0
                if all(k in keypoints_dict for k in [23, 25, 27]): # Hip, Knee, Ankle
                    knee_angle = calculate_angle_2d(
                        keypoints_dict[23], 
                        keypoints_dict[25], 
                        keypoints_dict[27]
                    )

                # ۷. ثبت فریم و زاویه در Neo4j
                try:
                    if neo4j_client._driver:
                        await neo4j_client.log_frame_analysis(
                            session_id=session_id,
                            frame_id=payload.frame_id,
                            knee_angle=round(knee_angle, 2),
                            is_valid=feedback.is_valid,
                            error_code=feedback.error_code
                        )
                except Exception as e:
                    logger.error(f"خطا در ثبت فریم در Neo4j: {e}")

                # ۸. ثبت فکت در Graphiti در صورت وجود خطای بیومکانیکی
                if not feedback.is_valid and feedback.error_code:
                    try:
                        if graphiti_client._graphiti:
                            await graphiti_client.log_biomechanical_fact(
                                user_id=user_id,
                                session_id=session_id,
                                error_code=feedback.error_code,
                                details=feedback.feedback_message
                            )
                    except Exception as e:
                        logger.error(f"خطا در ثبت فکت در Graphiti: {e}")

                # ۹. ارسال بازخورد آنی به کاربر
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