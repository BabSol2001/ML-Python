import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V dimensions: [batch_size, seq_len, d_k]
    """
    # ۱. استخراج بُعد کلیدها (d_k)
    d_k = Q.size(-1)
    
    # ۲. ضرب ماتریسی Q در ترانهاده K و تقسیم بر رادیکال d_k
    # K.transpose(-2, -1) دو بعد آخر ماتریس K را جابه‌جا می‌کند
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # ۳. اعمال ماسک (در صورت نیاز - مثلاً برای Masked Self-Attention در Decoder)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # ۴. اعمال Softmax روی آخرین بعد برای به دست آوردن وزن‌های توجه (Attention Weights)
    attention_weights = F.softmax(scores, dim=-1)
    
    # ۵. ضرب وزن‌های توجه در ماتریس V (مقادیر)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights