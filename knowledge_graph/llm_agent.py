import os
from typing import Optional
from openai import OpenAI

# اتصال به فایل اصلی
from networkx_triplets import KnowledgeGraph, build_sample_kg

class LLMGraphRAGAgent:
    def __init__(self, kg: KnowledgeGraph, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        اتصال به OpenAI یا مدل‌های محلی (Ollama / LM Studio)
        """
        self.kg = kg
        self.model_name = model_name
        
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY", "lm-studio/ollama")
        
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url  # مثلاً "http://localhost:11434/v1" برای Ollama
        )

    def answer_question(self, question: str, target_entity: str) -> str:
        """پاسخ بر اساس استخراج کانتکست گراف دانش"""
        if target_entity not in self.kg.entities:
            return f"خطا: موجودیت '{target_entity}' در گراف موجود نیست."

        affected_nodes = self.kg.analyze_impact(target_entity)
        all_nodes = affected_nodes.union({target_entity})
        sub_triplets = self.kg.get_subgraph_triplets(all_nodes)
        
        context_str = "\n".join([f"- {t.head} [{t.relation}] -> {t.tail}" for t in sub_triplets])

        system_prompt = """شما یک دستیار هوشمند تحلیل زنجیره تامین و مهندسی هستید.
پاسخ شما باید **دقیقاً و صرفاً** بر اساس حقایق موجود در کانتکست گراف دانش زیر باشد.
اگر پاسخی در کانتکست گراف وجود نداشت، از حدس زدن خودداری کنید."""

        user_prompt = f"""
کانتکست ساختاریافته از گراف دانش:
{context_str}

سوال کاربر:
{question}

لطفاً تحلیلی دقیق، روان و مدیریتی ارائه دهید:
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
            return f"[پاسخ آفلاین/شبکه‌ای]: بر اساس کانتکست گراف زیر:\n{context_str}\n\nتحلیل: بروز اختلال در {target_entity} مستقیماً موجودیت‌های پایین‌دست را متأثر می‌کند.\n(جزئیات خطا: {e})"

# ==========================================
# تست مستقل ایجنت LLM
# ==========================================
if __name__ == "__main__":
    # دریافت گراف از فایل اصلی
    kg = build_sample_kg()
    
    agent = LLMGraphRAGAgent(kg=kg)
    
    question = "اگر تامین‌کننده آلفا دچار بحران شود، دقیقا چه خطری سفارش شرکت بوئینگ را تهدید می‌کند؟"
    print(f"❓ سوال: {question}\n")
    
    response = agent.answer_question(question, target_entity="Supplier_Alpha")
    print(f"🤖 پاسخ ایجنت Graph-RAG:\n{response}")