import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# اضافه کردن یک پوشه عقب‌تر (Attention_Mechanism) به مسیرهای پایتون
sys.path.append(str(Path(__file__).resolve().parent.parent))

# وارد کردن لایه‌های اختصاصی و ماژول استخراج PDF
from multi_head_attention import MultiHeadAttention
from block import TransformerBlock
from pdf_extractor import prepare_dataset

print("--- شروع اجرای اسکریپت Deep Mini-GPT ---")

# ==========================================
# ۱. تنظیمات و ابرپارامترها (Hyperparameters)
# ارتقای ابعاد برای پشتیبانی از متن‌های ترکیبی
# ==========================================
batch_size = 32
block_size = 128         # افزایش طول زمینه (Context Window) برای درک کدها و متون
max_iters = 3500         # افزایش گام‌های آموزش برای رسیدن Loss به زیر 1.0
eval_interval = 300      # فاصله ارزیابی Loss
learning_rate = 5e-4     # نرخ یادگیری بهینه
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model = 128            # افزایش ابعاد امبدینگ برای پوشش کدهای پایتون و ریاضی
num_heads = 4
n_layer = 4              # ۴ لایه ترنسفورمر عمیق

# ==========================================
# ۲. داده‌های متنی و توکن‌ساز (از PDF یا TXT)
# ==========================================
input_file = "تفاوت یادگیری نظارت_شده و نظارت_نشده - Google Gemini.pdf" 

try:
    dataset_path = prepare_dataset(input_file)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    print(f"دیتاست با موفقیت بارگذاری شد. تعداد کل کاراکترها: {len(raw_text):,}")

except Exception as e:
    print(f"خطا در بارگذاری دیتاست: {e}")
    sys.exit(1)

# استخراج واژگان (حروف)
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
print(f"تعداد کاراکترهای منحصر‌به‌فرد (Vocab Size): {vocab_size}")

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

# ==========================================
# ۳. معماری مدل عمیق DeepMiniGPT
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

    def generate(self, idx, max_new_tokens, temperature=0.4):
        """تولید متن با پارامتر Temperature برای کنترل انسجام"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            
            # تقسیم بر Temperature برای کاهش تصادفی بودن پیش‌بینی‌ها
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# ۴. بارگذاری وزن‌ها یا شروع آموزش خودکار
# ==========================================
MODEL_PATH = 'deep_mini_gpt.pth'

model = DeepMiniGPT(
    vocab_size=vocab_size, 
    d_model=d_model, 
    num_heads=num_heads, 
    n_layer=n_layer, 
    max_seq_len=block_size
).to(device)

print(f"دستگاه پردازشی: {device}")
print(f"تعداد کل پارامترها با {n_layer} لایه ترنسفورمر: {sum(p.numel() for p in model.parameters()):,}\n")

weights_loaded = False

if os.path.exists(MODEL_PATH):
    print(f"فایل وزن‌های آماده ({MODEL_PATH}) پیدا شد!")
    print("در حال بارگذاری وزن‌ها...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("وزن‌های مدل با موفقیت بارگذاری شدند.\n")
        weights_loaded = True
    except Exception as e:
        print("ابعاد مدل با وزن‌های ذخیره‌شده قبلی هم‌خوانی ندارد (Vocab Size تغییر کرده است).")
        print("آموزش مدل از صفر شروع می‌شود...\n")

if not weights_loaded:
    print("شروع فرایند آموزش...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            model.eval()
            with torch.no_grad():
                _, train_loss = model(*get_batch('train'))
                _, val_loss = model(*get_batch('val'))
            print(f"گام {iter:4d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            model.train()

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nآموزش تمام شد. وزن‌های مدل در {MODEL_PATH} ذخیره شدند.\n")

# ==========================================
# ۵. حلقه چت تعاملی (Interactive Chat Loop)
# ==========================================
print("=" * 40)
print("   چت‌بات عمیق آماده است! (برای خروج exit بنویسید)")
print("=" * 40 + "\n")

model.eval()

while True:
    try:
        user_input = input("شما: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'خروج']:
            print("خداحافظ!")
            break
            
        if not user_input:
            continue

        encoded_input = [stoi[c] for c in user_input if c in stoi]
        
        if not encoded_input:
            print("چت‌بات: کاراکترهای ورودی شما در دیتاست آموزش داده شده وجود ندارد!\n")
            continue

        context = torch.tensor([encoded_input], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_ids = model.generate(context, max_new_tokens=120, temperature=0.4)[0].tolist()

        full_response = decode(generated_ids)
        print(f"چت‌بات:\n{full_response}\n")
        print("-" * 40)

    except KeyboardInterrupt:
        print("\nخداحافظ!")
        break