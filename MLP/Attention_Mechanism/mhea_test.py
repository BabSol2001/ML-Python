import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ۱. تابع Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


# ۲. کلاس Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model باید بر num_heads بخش‌پذیر باشد."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        output = self._combine_heads(attn_output)
        output = self.W_o(output)
        return output, attn_weights


# ۳. اجرای تست نمونه
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8

    # داده‌های نمونه
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # ساخت و اجرای مدل
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(Q, K, V)

    print("✅ تست موفقیت‌آمیز بود!")
    print("ابعاد خروجی:", output.shape)       # [2, 5, 512]
    print("ابعاد وزن‌ها:", weights.shape)      # [2, 8, 5, 5]