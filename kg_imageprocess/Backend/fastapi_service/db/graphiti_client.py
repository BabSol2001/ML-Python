from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from config import settings

class GraphitiClient:
    """
    مدیریت موتور گراف دانش زمان‌مند Graphiti برای حافظه بلندمدت بیومکانیکی کاربر
    """
    def __init__(self):
        self._graphiti: Optional[Graphiti] = None

    async def initialize(self) -> None:
        """
        مقداردهی اولیه موتور Graphiti و اتصال آن به Neo4j
        """
        if not self._graphiti:
            self._graphiti = Graphiti(
                uri=settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD
            )
            await self._graphiti.build_indices_and_constraints()

    async def close(self) -> None:
        """
        بستن اتصالات داخلی Graphiti
        """
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

    async def add_workout_episode(
        self,
        user_id: str,
        session_id: str,
        episode_body: str,
        source_description: str = "Workout Session Analysis",
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        ثبت یک اپیزود زمان‌مند از خلاصه تمرین کاربر در گراف حافظه.
        """
        if not self._graphiti:
            raise RuntimeError("موتور Graphiti هنوز مقداردهی نشده است. ابتدا initialize() را فراخوانی کنید.")

        ref_time = timestamp or datetime.now(timezone.utc)
        content = f"User ID: {user_id} | Session: {session_id} | {episode_body}"

        # استفاده از EpisodeType.text به عنوان مقدار معتبر source
        await self._graphiti.add_episode(
            name=f"workout_{session_id}",
            episode_body=content,
            source=EpisodeType.text,
            source_description=f"User {user_id} - {source_description}",
            reference_time=ref_time
        )

    async def search_user_memory(
        self,
        user_id: str,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        جستجو در حافظه زمان‌مند کاربر بر اساس پرامپت متنی (مورد استفاده در MS-RAG)
        """
        if not self._graphiti:
            raise RuntimeError("موتور Graphiti هنوز مقداردهی نشده است.")

        user_query = f"User {user_id}: {query}"
        results = await self._graphiti.search(query=user_query)
        
        memory_facts = []
        for result in results[:num_results]:
            memory_facts.append({
                "fact": getattr(result, "fact", getattr(result, "name", str(result))),
                "valid_at": str(getattr(result, "valid_at", "")),
                "invalid_at": str(getattr(result, "invalid_at", ""))
            })
        return memory_facts


# نمونه singleton برای استفاده در سراسر پروژه FastAPI
graphiti_client = GraphitiClient()