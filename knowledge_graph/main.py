# main.py
import os
from networkx_triplets import build_sample_kg, GraphRAGAgent
from llm_agent import LLMGraphRAGAgent

def run_pipeline():
    print("🚀 [Step 1/3] ساخت گراف دانش پایه...")
    kg = build_sample_kg()
    print(f"   -> تعداد گره‌ها: {len(kg.entities)} | تعداد روابط: {len(kg.triplets)}")

    print("\n📊 [Step 2/3] تولید داشبورد تعاملی گراف...")
    risk_source = "Supplier_Alpha"
    kg.visualize_interactive("smartbiz_pipeline_output.html", highlight_risk_from=risk_source)

    print(f"\n🧠 [Step 3/3] اجرای ایجنت Graph-RAG برای سناریوی بحران در '{risk_source}'...")
    agent = LLMGraphRAGAgent(kg=kg)
    
    question = f"اگر در {risk_source} اختلالی پیش بیاید، کدام قطعات و مشتریان نهایی آسیب می‌بینند؟"
    answer = agent.answer_question(question, target_entity=risk_source)
    
    print("\n" + "="*50)
    print(f"سوال: {question}")
    print("="*50)
    print(f"پاسخ ایجنت:\n{answer}")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()