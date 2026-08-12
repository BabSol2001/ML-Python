import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pdf_extractor_persian import prepare_dataset
from real_time_translator import RealTimeTranslator

print("=== راهکار ۳: ترکیب RAG اختصاصی با مدل پیش‌آموخته SLM (Qwen2.5-0.5B) ===")
translator = RealTimeTranslator()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ۱. بارگذاری SLM سبک
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"در حال بارگذاری مدل SLM: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

# ۲. بارگذاری داده و RAG
input_file = "Introduction to Machine Learning with Python OReilly.pdf"
dataset_path = prepare_dataset(input_file)
with open(dataset_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

class SLMRAGRetriever:
    def __init__(self, full_text):
        paragraphs = full_text.split('\n\n')
        self.chunks = [p.strip() for p in paragraphs if p.strip()]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.chunk_vectors).flatten()
        return self.chunks[sims.argmax()]

retriever = SLMRAGRetriever(raw_text)

# ۳. حلقه چت
while True:
    try:
        user_fa = input("\nشما (فارسی): ").strip()
        if user_fa.lower() in ['exit', 'quit']: break
        if not user_fa: continue

        user_en = translator.fa_to_en(user_fa)
        context_en = retriever.retrieve(user_en)
        print(f"[متن کتاب استخراج‌شده توسط RAG]:\n« {context_en[:200]}... »")

        # پرامپت استاندارد برای SLM
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer the query based strictly on the provided context."},
            {"role": "user", "content": f"Context:\n{context_en}\n\nQuestion: {user_en}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=150, temperature=0.2)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response_en = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\nچت‌بات (پاسخ هوشمند SLM به انگلیسی):\n{response_en}")
        print(f"\nچت‌بات (ترجمه فارسی):\n{translator.en_to_fa(response_en)}")
        print("-" * 50)
    except KeyboardInterrupt: break