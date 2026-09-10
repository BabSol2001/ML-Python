import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase  # استفاده از درایور بومی برای اجرای Cypher
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from config import settings

logger = logging.getLogger("GraphitiClient")
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("graphiti_core").setLevel(logging.ERROR)


class GraphitiClient:
    """
    مدیریت موتور گراف دانش زمان‌مند Graphiti برای حافظه بلندمدت بیومکانیکی کاربر
    """

    def __init__(self):
        self._graphiti: Optional[Graphiti] = None
        self._driver: Optional[Neo4jDriver] = None
        self._native_driver = None  # درایور اختصاصی Neo4j

    async def initialize(self) -> None:
        """
        مقداردهی اولیه موتور Graphiti و درایورهای Neo4j
        """
        os.environ.setdefault("OPENAI_API_KEY", "ollama")

        neo4j_uri = settings.NEO4J_URI.replace("localhost", "127.0.0.1")
        db_name = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")
        auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

        # ۱. درایور بومی برای کوئری‌های صریح Cypher
        if not self._native_driver:
            self._native_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=auth)

        # ۲. درایور مخصوص Graphiti
        if not self._driver:
            self._driver = Neo4jDriver(
                uri=neo4j_uri,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                database=db_name
            )

        if not self._graphiti:
            ollama_api_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"

            custom_openai_client = AsyncOpenAI(
                base_url=ollama_api_url,
                api_key="ollama",
                timeout=600.0
            )

            llm_config = LLMConfig(
                api_key="ollama",
                base_url=ollama_api_url,
                model=settings.OLLAMA_MODEL,
                small_model=settings.OLLAMA_MODEL
            )
            llm_client = OpenAIClient(config=llm_config, client=custom_openai_client)

            embedder_config = OpenAIEmbedderConfig(
                api_key="ollama",
                base_url=ollama_api_url,
                embedding_model="nomic-embed-text",
                embedding_dim=768
            )
            embedder_client = OpenAIEmbedder(config=embedder_config, client=custom_openai_client)

            self._graphiti = Graphiti(
                graph_driver=self._driver,
                llm_client=llm_client,
                embedder=embedder_client,
                cross_encoder=None
            )
            logger.info(f"✅ موتور Graphiti با موفقیت روی دیتابیس '{db_name}' مقداردهی شد.")

    async def close(self) -> None:
        """بستن اتصالات دیتابیس"""
        if self._native_driver:
            await self._native_driver.close()
            self._native_driver = None
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._graphiti = None

    async def add_workout_episode(
        self,
        user_id: str,
        session_id: str,
        episode_body: str,
        source_description: str = "Workout Session Analysis",
        timestamp: Optional[datetime] = None
    ) -> None:
        if not self._graphiti:
            raise RuntimeError("موتور Graphiti هنوز مقداردهی نشده است.")

        ref_time = timestamp or datetime.now(timezone.utc)
        content = f"Session: {session_id} | {episode_body}"

        await self._graphiti.add_episode(
            name=f"workout_{session_id}",
            episode_body=content,
            source=EpisodeType.text,
            source_description=f"User {user_id} - {source_description}",
            reference_time=ref_time,
            group_id=user_id
        )

    async def log_biomechanical_fact(self, user_id: str, session_id: str, error_code: str, details: str) -> None:
        fact_text = f"Biomechanical error detected: {error_code}. Details: {details}"
        await self.add_workout_episode(
            user_id=user_id,
            session_id=session_id,
            episode_body=fact_text,
            source_description="Realtime Pose Analysis Error"
        )

    async def search_user_memory(
        self,
        user_id: str,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        جستجو در حافظه گرافی کاربر
        """
        memory_facts = []

        # ۱. تلاش برای جستجوی سمانتیک با Graphiti
        if self._graphiti:
            try:
                search_response = await self._graphiti.search(query=query, group_ids=[user_id])
                edges = getattr(search_response, "edges", []) or getattr(search_response, "results", []) or (search_response if isinstance(search_response, list) else [])
                for item in edges[:num_results]:
                    fact_text = getattr(item, "fact", None) or getattr(item, "content", None) or getattr(item, "episode_body", None)
                    if fact_text and str(fact_text) not in [m["fact"] for m in memory_facts]:
                        memory_facts.append({
                            "fact": str(fact_text),
                            "valid_at": str(getattr(item, "valid_at", None) or getattr(item, "created_at", None))
                        })
            except Exception as e:
                logger.warning(f"⚠️ خطای جستجوی سمانتیک: {e}")

        # ۲. Fallback صریح Cypher با درایور بومی
        if not memory_facts and self._native_driver:
            try:
                db_name = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")
                cypher_query = """
                MATCH (u:User {id: $user_id})-[*1..2]-(n)
                WITH n,
                     CASE 
                        WHEN 'Frame' IN labels(n) THEN
                            "Frame Error: " + COALESCE(n.error_code, "Unknown Error") + 
                            " | Knee Angle: " + toString(COALESCE(n.knee_angle, "")) + 
                            " | Frame ID: " + toString(COALESCE(n.frame_id, ""))
                        WHEN 'WorkoutSession' IN labels(n) THEN
                            "Workout Session: " + COALESCE(n.exercise_name, "Exercise")
                        ELSE
                            COALESCE(n.fact, n.details, n.episode_body, n.content, n.name)
                     END AS fact_str,
                     COALESCE(n.timestamp, n.created_at, n.valid_at) AS valid_time
                WHERE fact_str IS NOT NULL AND fact_str <> ''
                RETURN DISTINCT fact_str AS content, valid_time
                ORDER BY valid_time DESC
                LIMIT $limit
                """
                
                records, _, _ = await self._native_driver.execute_query(
                    cypher_query, 
                    user_id=user_id,
                    limit=num_results,
                    database_=db_name
                )
                
                for rec in records:
                    content = rec["content"]
                    created_at = rec["valid_time"]
                    if content and content not in [m["fact"] for m in memory_facts]:
                        memory_facts.append({
                            "fact": str(content),
                            "valid_at": str(created_at) if created_at else None
                        })
            except Exception as e:
                logger.error(f"⚠️ خطای بازیابی مستقیم Cypher: {e}")

        return memory_facts


graphiti_client = GraphitiClient()