import json
import re
import time
from typing import List, Dict, cast
try:
    from typing import LiteralString  # Python 3.11+
except ImportError:
    from typing_extensions import LiteralString  # Python < 3.11

from pypdf import PdfReader
from openai import OpenAI, RateLimitError, APIError
from neo4j_manager import Neo4jKnowledgeGraphManager

# تایپ‌های مجاز تعریف‌شده در پرامپت
ALLOWED_TYPES = {
    "Technology", "Database", "Concept", "Tool", 
    "Language", "Person", "Organization", "Dataset", "DomainEntity"
}

class PDFGraphExtractor:
    def __init__(self, groq_api_key: str, neo4j_manager: Neo4jKnowledgeGraphManager, model: str = "llama-3.3-70b-versatile"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key
        )
        self.neo4j = neo4j_manager
        self.model = model

    def extract_text_chunks(self, pdf_path: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """۱. استخراج و تکه‌تکه‌سازی متن برای حفظ دقت مدل"""
        reader = PdfReader(pdf_path)
        full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        chunks = []
        for i in range(0, len(full_text), chunk_size - overlap):
            chunks.append(full_text[i:i + chunk_size])
        return chunks

    def extract_triplets_from_chunk(self, chunk: str, max_retries: int = 5, initial_delay: float = 4.0) -> List[Dict[str, str]]:
        """۲. استخراج سه‌تایی‌ها همراه با سیستم Retry و Exponential Backoff جهت مدیریت Rate Limit"""
        system_prompt = """تو یک متخصص برجسته استخراج گراف دانش (Knowledge Graph) هستی.
وظیفه تو استخراج دقیق موجودیت‌های اصلی و روابط معنی‌دار بین آن‌ها از متن است.

قوانین استخراج (بسیار حیاتی):
۱. **فقط تایپ‌های مجاز:** تایپ گره‌ها (head_type و tail_type) را دقیقاً و **فقط** از این لیست انتخاب کن:
   [Technology, Database, Concept, Tool, Language, Person, Organization, Dataset, DomainEntity]

۲. **استثنائات و خطوط قرمز (فیلتر نویز):**
   - هیچ‌گونه کد یا کوئری (مانند دستورات Cypher یا MATCH ...) را به عنوان موجودیت (Entity) استخراج نکن.
   - پسوند فایل‌ها (مانند csv. یا categories.csv) نباید گره باشند.
   - ویژگی‌ها و Properties (مانند orderID، orderDate، Name، Description) را به عنوان گره مجزا استخراج نکن.
   - روابط خود-ارجاعی (مانند Cypher -> Cypher) ایجاد نکن.

۳. **استانداردسازی نام‌ها:**
   - نام موجودیت‌ها را یکدست نگه دار (مثلاً همیشه "Neo4j" نه "Neo4j DB").
   - نام رابطه (relation) باید کوتاه، دقیق و به صورت UPPER_SNAKE_CASE باشد (مانند: USES, DEVELOPED_BY, PART_OF, INTEGRATES_WITH).

پاسخ را **دقیقاً** در قالب JSON زیر برگردان:
{
  "triplets": [
    {
      "head": "Neo4j",
      "head_type": "Database",
      "relation": "USES_QUERY_LANGUAGE",
      "tail": "Cypher",
      "tail_type": "Language"
    }
  ]
}"""

        user_prompt = f"متن جهت تحلیل:\n{chunk}"

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                raw_content = response.choices[0].message.content

                if not raw_content:
                    print("⚠️ پاسخ دریافت شده از مدل خالی (None) است.")
                    return []

                data = json.loads(raw_content)
                triplets = data.get("triplets", [])
                return triplets if isinstance(triplets, list) else []

            except RateLimitError as e:
                # محاسبه زمان انتظار به صورت تصاعدی (Exponential Backoff)
                delay = initial_delay * (2 ** attempt)
                print(f"⚠️ محدودیت Rate Limit (429) رخ داد. تلاش مجدد ({attempt + 1}/{max_retries}) پس از {delay:.1f} ثانیه...")
                time.sleep(delay)

            except APIError as e:
                print(f"❌ خطای API: {e}")
                break

            except Exception as e:
                print(f"❌ خطا در استخراج از تکه متن: {e}")
                break

        return []

    def _clean_type(self, entity_type: str) -> str:
        """تطبیق و صحت‌سنجی نوع موجودیت با لیست مجاز"""
        if not entity_type:
            return "Concept"
        cleaned = entity_type.strip()
        for allowed in ALLOWED_TYPES:
            if cleaned.lower() == allowed.lower():
                return allowed
        return "Concept"

    def _clean_relation(self, relation: str) -> str:
        """پالایش نام روابط جهت جلوگیری از خطاهای Cypher Syntax در Neo4j"""
        if not relation:
            return "RELATED_TO"
        cleaned = re.sub(r'[^A-Z0-9_]', '_', relation.strip().upper())
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        return cleaned or "RELATED_TO"

    def filter_and_post_process(self, triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """۳. فیلتر کردن موارد تکراری و حذف نویزهای احتمالی باقی‌مانده"""
        cleaned_triplets = []
        seen = set()

        for t in triplets:
            head = t.get("head", "").strip()
            tail = t.get("tail", "").strip()
            relation = t.get("relation", "").strip()
            head_type = t.get("head_type", "").strip()
            tail_type = t.get("tail_type", "").strip()

            if not head or not tail or head.lower() == tail.lower():
                continue
            if "MATCH " in head or "CREATE " in head or head.endswith(".csv"):
                continue
            if "MATCH " in tail or "CREATE " in tail or tail.endswith(".csv"):
                continue

            unique_key = (head.lower(), relation.lower(), tail.lower())
            if unique_key in seen:
                continue

            seen.add(unique_key)
            cleaned_triplets.append({
                "head": head,
                "head_type": self._clean_type(head_type),
                "relation": self._clean_relation(relation),
                "tail": tail,
                "tail_type": self._clean_type(tail_type)
            })

        return cleaned_triplets

    def save_triplets_to_neo4j(self, triplets: List[Dict[str, str]]):
        """۴. تزریق امن و ساختاریافته به Neo4j"""
        if not triplets:
            print("⚠️ هیچ سه‌تایی معتبری برای ذخیره وجود ندارد.")
            return

        with self.neo4j.driver.session() as session:
            for t in triplets:
                raw_query_string = f"""
                MERGE (h:`{t['head_type']}` {{name: $head_name}})
                MERGE (t:`{t['tail_type']}` {{name: $tail_name}})
                MERGE (h)-[r:`{t['relation']}`]->(t)
                """
                
                cypher_query = cast(LiteralString, raw_query_string)

                try:
                    session.run(cypher_query, head_name=t['head'], tail_name=t['tail'])
                except Exception as e:
                    print(f"❌ خطا در ثبت رابطه [{t['head']}] -> [{t['tail']}]: {e}")

    def process_pdf_file(self, pdf_path: str, chunk_delay: float = 1.0):
        """مدیریت زنجیره کامل پردازش PDF"""
        print("۱. در حال تکه‌تکه‌سازی متن PDF...")
        chunks = self.extract_text_chunks(pdf_path)
        print(f"➜ تعداد {len(chunks)} تکه متن ایجاد شد.")
        
        raw_triplets = []
        for index, chunk in enumerate(chunks):
            print(f"۲. پردازش بخش {index + 1} از {len(chunks)}...")
            triplets = self.extract_triplets_from_chunk(chunk)
            raw_triplets.extend(triplets)
            
            # ایجاد تاخیر بین درخواست‌ها جهت عدم عبور از سقف RPM/TPM
            if chunk_delay > 0:
                time.sleep(chunk_delay)
            
        print("۳. در حال پالایش و حذف نویزها...")
        cleaned_triplets = self.filter_and_post_process(raw_triplets)
        print(f"➜ تعداد {len(cleaned_triplets)} سه‌تایی معتبر و خالص باقی ماند.")
        
        print("۴. در حال تزریق داده‌ها به Neo4j...")
        self.save_triplets_to_neo4j(cleaned_triplets)
        print("✅ گراف دانش با موفقیت ساخته شد.")
        return cleaned_triplets