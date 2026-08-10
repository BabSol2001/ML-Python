import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # ۱. لایه‌های نُرمال‌سازی
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # ۲. مکانیزم توجه چندسر (کلاس اختصاصی خودتان)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # ۳. شبکه پیش‌سو (Feed Forward)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # استفاده از GELU به‌جای ReLU (مانند GPT)
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # اتصال باقی‌مانده اول: x + Attention(LN1(x))
        norm_x1 = self.ln1(x)
        attn_out, _ = self.attention(Q=norm_x1, K=norm_x1, V=norm_x1, mask=mask)
        x = x + attn_out  # Residual Connection 1
        
        # اتصال باقی‌مانده دوم: x + FFN(LN2(x))
        norm_x2 = self.ln2(x)
        ff_out = self.feed_forward(norm_x2)
        x = x + ff_out    # Residual Connection 2
        
        return x