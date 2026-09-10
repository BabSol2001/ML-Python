import uuid
import logging
from typing import Any, Dict, List, Optional
from typing_extensions import LiteralString
from neo4j import AsyncGraphDatabase, AsyncDriver, Query
from config import settings

logger = logging.getLogger("Neo4jClient")

class Neo4jClient:
    """
    مدیریت Connection Pool و کوئری‌های غیرهمگام به Neo4j
    """
    def __init__(self):
        self._driver: Optional[AsyncDriver] = None
        # تنظیم دیتابیس هدف (در صورت عدم وجود در کانفیگ، kgimageprocessdb جایگزین می‌شود)
        self._db_name: str = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")

    async def connect(self) -> None:
        """برقراری اتصال غیرهمگام به Neo4j هنگام شروع برنامه"""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            logger.info(f"✅ اتصال به Neo4j روی دیتابیس '{self._db_name}' برقرار شد.")

    async def close(self) -> None:
        """بستن Connection Pool هنگام خاموش شدن سرور"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("❌ اتصال Neo4j بسته شد.")

    async def execute_query(
        self, 
        query: LiteralString | Query | str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        اجرای یک کوئری Cypher روی دیتابیس اختصاصی
        """
        if not self._driver:
            raise RuntimeError("اتصال Neo4j هنوز برقرار نشده است. ابتدا تابع connect() را فراخوانی کنید.")

        query_obj = query if isinstance(query, Query) else Query(str(query))  # type: ignore

        # اصلاح اصلی: مشخص کردن دیتابیس هدف در session
        async with self._driver.session(database=self._db_name) as session:
            result = await session.run(query_obj, parameters or {})
            records = await result.data()
            return records

    async def check_connection(self) -> bool:
        """تست سریع سلامت اتصال به گراف"""
        try:
            records = await self.execute_query("RETURN 1 AS test")
            return len(records) > 0 and records[0].get("test") == 1
        except Exception as e:
            logger.error(f"خطا در اتصال به Neo4j ({self._db_name}): {e}")
            return False

    # =========================================================================
    # متدهای بیومکانیکی (Domain Methods)
    # =========================================================================

    async def create_session_node(self, user_id: str, exercise_name: str, session_uuid: Optional[str] = None) -> str:
        """
        ایجاد گره Root برای جلسه تمرینی جدید
        """
        session_id = session_uuid or str(uuid.uuid4())
        cypher = """
        MERGE (u:User {id: $user_id})
        CREATE (s:WorkoutSession {
            id: $session_id,
            exercise_name: $exercise_name,
            created_at: datetime()
        })
        CREATE (u)-[:PERFORMED]->(s)
        RETURN s.id AS session_id
        """
        await self.execute_query(cypher, {
            "user_id": user_id,
            "session_id": session_id,
            "exercise_name": exercise_name
        })
        logger.info(f"📊 گره WorkoutSession با شناسه {session_id} در دیتابیس {self._db_name} ثبت شد.")
        return session_id

    async def log_frame_analysis(
        self, 
        session_id: str, 
        frame_id: int, 
        knee_angle: float, 
        is_valid: bool, 
        error_code: Optional[str] = None
    ) -> None:
        """
        ثبت فریم تحلیل‌شده و متصل کردن آن به جلسه تمرینی در Neo4j
        """
        cypher = """
        MATCH (s:WorkoutSession {id: $session_id})
        CREATE (f:Frame {
            frame_id: $frame_id,
            knee_angle: $knee_angle,
            is_valid: $is_valid,
            error_code: $error_code,
            timestamp: datetime()
        })
        CREATE (s)-[:HAS_FRAME]->(f)
        """
        await self.execute_query(cypher, {
            "session_id": session_id,
            "frame_id": frame_id,
            "knee_angle": knee_angle,
            "is_valid": is_valid,
            "error_code": error_code or "NONE"
        })


neo4j_client = Neo4jClient()