import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# اضافه کردن مسیر پروژه
sys.path.append(str(Path(__file__).resolve().parent.parent))

from multi_head_attention import MultiHeadAttention
from block import TransformerBlock
from pdf_extractor_persian import prepare_dataset

print("--- اجرای چت‌بات مجهز به سیستم RAG (Retrieval-Augmented Generation) ---")

from real_time_translator import RealTimeTranslator

# مقداردهی اولیه مترجم
translator = RealTimeTranslator()

# ==========================================
# ۱. تنظیمات و ابرپارامترها (Hyperparameters)
# ==========================================
batch_size = 32
block_size = 128
max_iters = 3500
eval_interval = 300
eval_iters = 50
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model = 128
num_heads = 4
n_layer = 4

input_file = "تفاوت یادگیری نظارت_شده و نظارت_نشده - Google Gemini.pdf"
MODEL_PATH = 'deep_mini_gpt.pth'

# ==========================================
# ۲. آماده‌سازی دیتاست و توکن‌ساز
# ==========================================
try:
    dataset_path = prepare_dataset(input_file)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"دیتاست با موفقیت بارگذاری شد. تعداد کل کاراکترها: {len(raw_text):,}")
except Exception as e:
    print(f"خطا در بارگذاری دیتاست: {e}")
    sys.exit(1)

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# ۳. پیاده‌سازی ماژول RAG اصلاح‌شده (TF-IDF + Vector Search)
# ==========================================
class AdvancedRAGRetriever:
    """تکه‌تکه‌سازی هوشمند متون PDF و بازیابی بر اساس تشابه کسینوسی TF-IDF"""
    def __init__(self, full_text, chunk_size=800, chunk_overlap=150):
        self.chunks = []
        
        # ۱. تکه‌تکه‌سازی بر اساس پاراگراف‌ها یا طول استاندارد
        paragraphs = full_text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += " " + para
            else:
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                current_chunk = para
                
        if current_chunk:
            self.chunks.append(current_chunk.strip())
            
        print(f"ماژول RAG فعال شد. تعداد تکه‌های متنی (Chunks): {len(self.chunks)}")

        # ۲. ساخت ماتریس TF-IDF برای تمام چانک‌ها
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query, top_k=1, min_similarity=0.1):
        """یافتن مرتبط‌ترین چانک بر اساس تشابه برداری TF-IDF"""
        if not query.strip() or not self.chunks:
            return ""

        # تبدیل سوال کاربر به بردار TF-IDF
        query_vector = self.vectorizer.transform([query])
        
        # محاسبه میزان تشابه کسینوسی بین سوال و تمام چانک‌ها
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # یافتن بهترین اندیس
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        # اگر هیچ چانک مرتبطی پیدا نشد (کمتر از آستانه تشابه)، چانک اشتباه بازنگردان
        if best_score < min_similarity:
            print(f"[RAG Warning]: هیچ متن مرتبطی پیدا نشد (میزان تشابه: {best_score:.2f})")
            return "No relevant context found in the document."

        print(f"[RAG Match Score]: {best_score:.4f} | Chunk Index: {best_idx}")
        return self.chunks[best_idx]

# جایگزینی ریتریور جدید
retriever = AdvancedRAGRetriever(raw_text)

# ==========================================
# ۴. معماری مدل DeepMiniGPT
# ==========================================
class DeepMiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, n_layer, max_seq_len):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.tok_embedding(idx) + self.pos_embedding(positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.3):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# تابع ارزیابی خطای مدل
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ==========================================
# ۵. بارگذاری مدل و حلقه آموزش (در صورت نیاز)
# ==========================================
model = DeepMiniGPT(
    vocab_size=vocab_size, 
    d_model=d_model, 
    num_heads=num_heads, 
    n_layer=n_layer, 
    max_seq_len=block_size
).to(device)

train_needed = False

if os.path.exists(MODEL_PATH):
    print(f"\nفایل وزن‌ها ({MODEL_PATH}) پیدا شد.")
    try:
        user_choice = input("آیا می‌خواهید مدل را مجدداً از ابتدا آموزش دهید؟ (y/n) [پیش‌فرض: n]: ").strip().lower()
    except EOFError:
        user_choice = 'n'
        
    if user_choice == 'y':
        train_needed = True
    else:
        print("در حال بارگذاری وزن‌های موجود...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            print("وزن‌ها با موفقیت بارگذاری شدند.\n")
        except Exception as e:
            print(f"خطا در بارگذاری فایل وزن‌ها: {e}\nفرایند آموزش مجدد آغاز می‌شود...")
            train_needed = True
else:
    print(f"\nفایل وزن‌ها ({MODEL_PATH}) پیدا نشد.")
    train_needed = True

if train_needed:
    print("\n" + "=" * 50)
    print(f"شروع آموزش مدل Mini-GPT روی دستگاه: {device.upper()}")
    print("=" * 50)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for iter_num in range(1, max_iters + 1):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            losses = estimate_loss(model)
            print(f"تکرار {iter_num}/{max_iters} -> خطای آموزش (Train Loss): {losses['train']:.4f} | خطای اعتبارسنجی (Val Loss): {losses['val']:.4f}")

    # ذخیره‌سازی وزن‌های جدید
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nآموزش کامل شد! وزن‌ها در فایل '{MODEL_PATH}' ذخیره گردیدند.\n")

# ==========================================
# ۶. حلقه چت RAG تعاملی (اصلاح‌شده)
# ==========================================
print("=" * 50)
print("   چت‌بات RAG آماده پاسخ‌گویی بر اساس PDF است!")
print("=" * 50 + "\n")

model.eval()

while True:
    try:
        user_input_fa = input("شما (فارسی): ").strip()
        
        if user_input_fa.lower() in ['exit', 'quit', 'خروج']:
            print("خداحافظ!")
            break
            
        if not user_input_fa:
            continue

        # ۱. ترجمه سوال کاربر به انگلیسی
        user_input_en = translator.fa_to_en(user_input_fa)
        print(f"\n[سوال ترجمه‌شده به انگلیسی]: {user_input_en}")

        # ۲. استخراج چانک مرتبط توسط RAG
        relevant_context_en = retriever.retrieve(user_input_en)
        print(f"[بخش مرتبط پیدا شده در PDF به انگلیسی]:\n« {relevant_context_en[:150]}... »\n")

        # ۳. ساخت پرامپت استاندارد برای تولید پاسخ
        # استفاده از خروجی مستقیم RAG اگر مدل توان تولید متن پیچیده ندارد
        clean_context = relevant_context_en.replace('\n', ' ')
        
        # ساخت ورودی کوتاه‌شده برای مدل
        prompt = f"Context: {clean_context[:100]}\nQ: {user_input_en}\nAnswer:"
        
        encoded_input = [stoi[c] for c in prompt if c in stoi]
        if not encoded_input:
            print("کاراکترهای ورودی پشتیبانی نمی‌شوند.\n")
            continue

        encoded_input = encoded_input[-block_size:]
        context_tensor = torch.tensor([encoded_input], dtype=torch.long, device=device)

        # ۴. تولید متن با هایپرپارامترهای کنترل‌شده
        with torch.no_grad():
            # کاهش temperature به 0.2 جهت کاهش هذیان‌گویی
            generated_ids = model.generate(context_tensor, max_new_tokens=80, temperature=0.2)[0].tolist()

        full_generated_text = decode(generated_ids)
        
        # جداسازی بخش پاسخ از پرامپت اصلی
        if "Answer:" in full_generated_text:
            answer_only_en = full_generated_text.split("Answer:")[-1].strip()
        else:
            answer_only_en = full_generated_text[len(prompt):].strip()

        # ۵. ترجمه و نمایش پاسخ
        answer_fa = translator.en_to_fa(answer_only_en)
        
        print(f"چت‌بات (پاسخ اصلی به انگلیسی): {answer_only_en}")
        print(f"چت‌بات (ترجمه فارسی):\n{answer_fa}\n")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nخداحافظ!")
        break