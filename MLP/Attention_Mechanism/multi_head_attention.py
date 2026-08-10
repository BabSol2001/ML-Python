import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    تابع اصلی شما - روی تنسورهای ۳ بعدی و ۴ بعدی به‌طور یکسان کار می‌کند.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model باید بر num_heads بخش‌پذیر باشد."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # ۱. لایه‌های خطی برای نگاشت Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # لایه خطی خروجی (Output Projection)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        """
        تغییر شکل تنسور از [batch_size, seq_len, d_model]
        به [batch_size, num_heads, seq_len, d_k] جهت محاسبات موازی
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        """
        تغییر شکل معکوس از [batch_size, num_heads, seq_len, d_k]
        به [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # گام ۱: اعمال لایه‌های خطی
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # گام ۲: شکستن ابعاد به چند سر (Split Heads)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # تنظیم ابعاد ماسک برای ۴ بعدی شدن (در صورت وجود)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # گام ۳: فراخوانی تابع Scaled Dot-Product Attention روی تمام سرها به‌صورت همزمان
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # گام ۴: چسباندن خروجی سرها به یکدیگر
        output = self._combine_heads(attn_output)

        # گام ۵: لایه خطی نهایی
        output = self.W_o(output)

        return output, attn_weights