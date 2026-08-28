from typing import Any, Dict, List, Optional
from typing_extensions import LiteralString  # برای پشتیبانی در پایتون‌های قدیمی‌تر یا typing رسمی
from neo4j import AsyncGraphDatabase, AsyncDriver, Query
from config import settings

class Neo4jClient:
    """
    مدیریت Connection Pool و کوئری‌های غیرهمگام به Neo4j
    """
    def __init__(self):
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """برقراری اتصال غیرهمگام به Neo4j هنگام شروع برنامه"""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )

    async def close(self) -> None:
        """بستن Connection Pool هنگام خاموش شدن سرور"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def execute_query(
        self, 
        query: LiteralString | Query | str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        اجرای یک کوئری Cypher و بازگرداندن نتایج به صورت لیستی از دیکشنری‌ها
        """
        if not self._driver:
            raise RuntimeError("اتصال Neo4j هنوز برقرار نشده است. ابتدا تابع connect() را فراخوانی کنید.")

        # تبدیل متغیر رشته‌ای معمولی به شیء Query رسمی درایور Neo4j
        query_obj = query if isinstance(query, Query) else Query(str(query)) # type: ignore

        async with self._driver.session() as session:
            result = await session.run(query_obj, parameters or {})
            records = await result.data()
            return records

    async def check_connection(self) -> bool:
        """تست سریع سلامت اتصال به گراف"""
        try:
            records = await self.execute_query("RETURN 1 AS test")
            return len(records) > 0 and records[0].get("test") == 1
        except Exception as e:
            print(f"خطا در اتصال به Neo4j: {e}")
            return False

# نمونه ایجادشده برای استفاده در کل برنامه (Singleton Pattern)
neo4j_client = Neo4jClient()