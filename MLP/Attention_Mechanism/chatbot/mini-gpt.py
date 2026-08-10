import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# اضافه کردن یک پوشه عقب‌تر (Attention_Mechanism) به مسیرهای پایتون
sys.path.append(str(Path(__file__).resolve().parent.parent))

# وارد کردن لایه اتنشن خودتان
from multi_head_attention import MultiHeadAttention

print("--- شروع اجرای اسکریپت Mini-GPT ---")

# ==========================================
# ۱. تنظیمات و ابرپارامترها (Hyperparameters)
# ==========================================
batch_size = 16
block_size = 32          # حداکثر طول متن (seq_len)
max_iters = 500          # تعداد گام‌های آموزش
eval_interval = 100      # فاصله ارزیابی Loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model = 64
num_heads = 4

# ==========================================
# ۲. داده‌های متنی و توکن‌ساز کاراکتری
# ==========================================
raw_text = """
سلام! چطور می‌توانم به شما کمک کنم؟
امروز یک روز عالی برای یادگیری هوش مصنوعی و مدل‌های زبانی است.
ترنسفورمرها و سیستم‌های اتنشن موتور اصلی مدل‌های هوش مصنوعی مدرن هستند.
سیلیکون نیترید و ماشین‌کاری با کمک لیزر پروژه‌های پیشرفته مهندسی و هوش مصنوعی هستند.
برای ساخت یک چت‌بات خوب باید مکانیزم توجه را به درستی درک کرد.
""" * 10

# استخراج واژگان (حروف)
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

# ==========================================
# ۳. ساختار کامل مدل MiniGPTChatbot
# ==========================================
class MiniGPTChatbot(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # لایه MultiHeadAttention اختصاصی شما
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        
        # ۱. تبدیل توکن‌ها و موقعیت‌ها به بردار
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.tok_embedding(idx) + self.pos_embedding(positions)
        
        # ۲. ساخت ماسک علّی (Causal Mask)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
        
        # ۳. اعمال توجه چندسر
        attn_out, _ = self.attention(Q=x, K=x, V=x, mask=mask)
        x = x + attn_out  # Residual Connection
        
        # ۴. لایه Feed Forward
        ff_out = self.feed_forward(x)
        x = x + ff_out    # Residual Connection
        
        # ۵. پیش‌بینی کلمه بعدی
        logits = self.lm_head(x)
        
        # ۶. محاسبه Loss در صورت وجود Target
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """حلقه تولید متن کلمه به کلمه"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # برش متن اگر از طول مجاز بیشتر شد
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # نمونه‌گیری بر اساس احتمال (تولید طبیعی‌تر نسبت به argmax)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# ۴. مقداردهی اولیه و آموزش
# ==========================================
model = MiniGPTChatbot(
    vocab_size=vocab_size, 
    d_model=d_model, 
    num_heads=num_heads, 
    max_seq_len=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"دستگاه پردازشی: {device}")
print(f"تعداد پارامترهای مدل: {sum(p.numel() for p in model.parameters()):,}\n")

print("--- شروع فرایند آموزش ---")
model.train()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        with torch.no_grad():
            _, train_loss = model(*get_batch('train'))
            _, val_loss = model(*get_batch('val'))
        print(f"گام {iter:3d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        model.train()

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==========================================
# ۵. تست تولید متن توسط چت‌بات
# ==========================================
print("\n=== تست خروجی مدل ===")
model.eval()
prompt = "سلام"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

generated_ids = model.generate(context, max_new_tokens=80)[0].tolist()
print("متن تولید شده توسط مدل:")
print("-" * 30)
print(decode(generated_ids))
print("-" * 30)