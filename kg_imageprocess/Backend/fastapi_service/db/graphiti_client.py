import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from config import settings

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("graphiti_core").setLevel(logging.ERROR)


class GraphitiClient:
    """
    مدیریت موتور گراف دانش زمان‌مند Graphiti برای حافظه بلندمدت بیومکانیکی کاربر
    """

    def __init__(self):
        self._graphiti: Optional[Graphiti] = None
        self._driver: Optional[Neo4jDriver] = None

    async def initialize(self) -> None:
        """
        مقداردهی اولیه موتور Graphiti با دیتابیس اختصاصی Neo4j و سرویس Ollama
        """
        if not self._graphiti:
            ollama_api_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"

            # ۱. ساخت کلاینت AsyncOpenAI با تایم‌اوت ۶۰۰ ثانیه‌ای (۱۰ دقیقه)
            custom_openai_client = AsyncOpenAI(
                base_url=ollama_api_url,
                api_key="ollama",
                timeout=600.0
            )

            # ۲. ساخت کانفیگ و کلاینت LLM
            llm_config = LLMConfig(
                api_key="ollama",
                base_url=ollama_api_url,
                model=settings.OLLAMA_MODEL,
                small_model=settings.OLLAMA_MODEL
            )
            llm_client = OpenAIClient(config=llm_config, client=custom_openai_client)

            # ۳. ساخت کانفیگ و کلاینت Embedder
            embedder_config = OpenAIEmbedderConfig(
                api_key="ollama",
                base_url=ollama_api_url,
                embedding_model="nomic-embed-text",
                embedding_dim=768
            )
            embedder_client = OpenAIEmbedder(config=embedder_config, client=custom_openai_client)

            # ۴. تنظیمات دیتابیس Neo4j
            neo4j_uri = settings.NEO4J_URI.replace("localhost", "127.0.0.1")
            db_name = getattr(settings, "NEO4J_DATABASE", "kgimageprocessdb")

            self._driver = Neo4jDriver(
                uri=neo4j_uri,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                database=db_name
            )

            self._graphiti = Graphiti(
                graph_driver=self._driver,
                llm_client=llm_client,
                embedder=embedder_client,
                cross_encoder=None
            )
            
            try:
                await self._graphiti.build_indices_and_constraints()
            except Exception:
                pass

    async def close(self) -> None:
        """بستن اتصالات دیتابیس"""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
            self._driver = None

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

    async def search_user_memory(
        self,
        user_id: str,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        جستجو در حافظه گرافی کاربر با پشتیبانی از ساختار جدید Edgeهای Graphiti و پشتیبان جامع Cypher
        """
        if not self._graphiti or not self._driver:
            raise RuntimeError("موتور Graphiti یا درایور Neo4j مقداردهی نشده است.")

        memory_facts = []

        # ۱. تلاش برای جستجو از طریق API استاندارد Graphiti
        try:
            search_response = await self._graphiti.search(query=query, group_ids=[user_id])
            
            edges = (
                getattr(search_response, "edges", None)
                or getattr(search_response, "results", None)
                or (search_response if isinstance(search_response, list) else [])
            )
            
            for item in edges[:num_results]:
                fact_text = (
                    getattr(item, "fact", None)
                    or getattr(item, "content", None)
                    or getattr(item, "body", None)
                    or getattr(item, "name", None)
                    or getattr(item, "episode_body", None)
                    or str(item)
                )
                if fact_text and fact_text not in [m["fact"] for m in memory_facts]:
                    memory_facts.append({
                        "fact": fact_text,
                        "valid_at": str(getattr(item, "valid_at", "")),
                        "invalid_at": str(getattr(item, "invalid_at", ""))
                    })
        except Exception as e:
            print(f"⚠️ خطای جستجوی گراف Graphiti: {e}")

        # ۲. Fallback صریح و قطعی با Cypher (پوشش تمام برچسب‌ها و پروپرتی‌های ممکن در Graphiti)
        if not memory_facts:
            try:
                cypher_query = """
                MATCH (n)
                WHERE n:Episodic OR n:EpisodicNode OR n:Entity OR n:EntityNode OR n:Fact OR n:Episode
                WITH n, COALESCE(n.body, n.content, n.fact, n.episode_body, n.name, n.source_description) AS text_val
                WHERE text_val IS NOT NULL AND text_val <> ''
                RETURN text_val AS content, 
                       COALESCE(n.created_at, n.valid_at) AS created_at
                ORDER BY created_at DESC
                LIMIT $limit
                """
                records, _, _ = await self._driver.execute_query(
                    cypher_query, 
                    limit=num_results
                )
                
                for rec in records:
                    content = rec["content"] if "content" in rec else None
                    created_at = rec["created_at"] if "created_at" in rec else None
                    
                    if content and str(content) not in [m["fact"] for m in memory_facts]:
                        memory_facts.append({
                            "fact": str(content),
                            "valid_at": str(created_at) if created_at else "",
                            "invalid_at": None
                        })
            except Exception as e:
                print(f"⚠️ خطای بازیابی مستقیم Cypher: {e}")

        return memory_facts

graphiti_client = GraphitiClient()