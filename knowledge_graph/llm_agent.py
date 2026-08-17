import os
from typing import Optional, Union, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# وارد کردن کلاس‌های گراف قدیمی در صورت وجود
try:
    from networkx_triplets import KnowledgeGraph, build_sample_kg
except ImportError:
    KnowledgeGraph = None

class LLMGraphRAGAgent:
    def __init__(
        self, 
        kg: Optional[Any] = None, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        model_name: Optional[str] = None
    ):
        """
        اتصال هوشمند به Groq ،OpenAI یا مدل‌های محلی
        پشتیبانی از هر دو نوع گراف: Neo4jKnowledgeGraphManager و KnowledgeGraph
        """
        self.kg = kg
        
        # ۱. تعیین API Key (اولیت: ورودی تابع > فایل .env > مقدار پیش‌فرض)
        resolved_api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "lm-studio/ollama")
        
        # ۲. تعیین Base URL (پیش‌فرض برای Groq)
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        
        # ۳. تعیین مدل (پیش‌فرض مدل llama-3.3-70b-versatile در Groq)
        self.model_name = model_name or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url
        )

    def answer_question_with_context(self, question: str, context: str) -> str:
        """
        پاسخ‌گویی بر اساس کانتکست استخراج‌شده از گراف دانش
        """
        system_prompt = """شما یک دستیار هوشمند Graph-RAG هستید.
مستقیماً و بدون مقدمه‌چینی یا اشاره به عدم وجود کانتکست، بر اساس روابط زیر و دانش گرافی به سوال کاربر پاسخ دهید."""

        user_prompt = f"""
کانتکست ساختاریافته از گراف دانش:
{context}

سوال کاربر:
{question}

لطفاً تحلیلی دقیق و کاربردی ارائه دهید:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            return content if content is not None else "پاسخی از مدل دریافت نشد."
        
        except Exception as e:
            return f"[پاسخ آفلاین/خطای شبکه]: کانتکست گراف استخراج‌شده:\n{context}\n\n(جزئیات خطا: {e})"

    def answer_question(self, question: str, target_entity: str) -> str:
        """
        پاسخ‌گویی مستقیم بر اساس یک entity خاص (سازگار با Neo4j و NetworkX)
        """
        context_lines: List[str] = []

        # ۱. استخراج اطلاعات در صورت اتصال به Neo4j
        # ۱. استخراج اطلاعات در صورت اتصال به Neo4j
        if self.kg and hasattr(self.kg, 'get_entity_context'):
            triplets = self.kg.get_entity_context(target_entity)
            print(f"\n[سیستم] تعداد روابط یافت‌شده برای '{target_entity}' از دیتابیس: {len(triplets)}")
            
            for t in triplets:
                line = f"- {t['head']} [{t['relation']}] -> {t['tail']}"
                context_lines.append(line)
                print(f"[گراف] {line}") # چاپ روابط در ترمینال برای اطمینان

        # ۲. استخراج اطلاعات در صورت استفاده از گراف NetworkX (کد قدیمی)
        elif self.kg and hasattr(self.kg, 'entities') and target_entity in self.kg.entities:
            affected_nodes = self.kg.analyze_impact(target_entity)
            all_nodes = affected_nodes.union({target_entity})
            sub_triplets = self.kg.get_subgraph_triplets(all_nodes)
            for t in sub_triplets:
                context_lines.append(f"- {t.head} [{t.relation}] -> {t.tail}")

        # ۳. ساخت کانتکست نهایی
        if context_lines:
            context_str = "\n".join(context_lines)
        else:
            context_str = f"موجودیت '{target_entity}' روابط مستقیمی در گراف ندارد یا یافت نشد."

        return self.answer_question_with_context(question, context=context_str)


# ==========================================
# تست مستقل ایجنت LLM
# ==========================================
if __name__ == "__main__":
    agent = LLMGraphRAGAgent()
    
    test_question = "زبان Cypher چه نقشی در اکوسیستم Neo4j دارد؟"
    print(f"❓ سوال: {test_question}\n")
    
    res = agent.answer_question(test_question, target_entity="Cypher")
    print(f"🤖 پاسخ ایجنت Graph-RAG با Groq:\n{res}")