import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# کدهای شما (در صورت مجزا بودن فایل‌ها می‌توانید آن‌ها را import کنید)
# =====================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
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


# =====================================================================
# مجموعه تست‌ها (Test Suite)
# =====================================================================

class TestScaledDotProductAttention(unittest.TestCase):
    """تست‌های مربوط به تابع Scaled Dot-Product Attention"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.d_k = 64
        
        self.Q = torch.randn(self.batch_size, self.seq_len, self.d_k)
        self.K = torch.randn(self.batch_size, self.seq_len, self.d_k)
        self.V = torch.randn(self.batch_size, self.seq_len, self.d_k)

    def test_output_and_weights_shape(self):
        """۱. بررسی ابعاد خروجی و وزن‌های توجه"""
        output, attn_weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_k))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.seq_len, self.seq_len))

    def test_softmax_sum_equals_one(self):
        """۲. بررسی اینکه مجموع احتمالات Softmax روی بعد آخر برابر با ۱ باشد"""
        _, attn_weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        row_sums = attn_weights.sum(dim=-1)
        expected_ones = torch.ones_like(row_sums)
        
        self.assertTrue(torch.allclose(row_sums, expected_ones, atol=1e-5))

    def test_masking_effect(self):
        """۳. بررسی صفر شدن وزن‌های توجه در موقعیت‌های ماسک‌شده"""
        # ساخت یک ماسک مثلثی پایین (Causal Mask)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        _, attn_weights = scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)
        
        # بررسی صفر بودن عناصر بالای قطر اصلی
        upper_triangle = torch.triu(attn_weights, diagonal=1)
        expected_zeros = torch.zeros_like(upper_triangle)
        
        self.assertTrue(torch.allclose(upper_triangle, expected_zeros, atol=1e-6))


class TestMultiHeadAttention(unittest.TestCase):
    """تست‌های مربوط به کلاس MultiHeadAttention"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 6
        self.d_model = 512
        self.num_heads = 8
        
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.Q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.K = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.V = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_forward_shape(self):
        """۱. بررسی ابعاد خروجی MultiHeadAttention"""
        output, attn_weights = self.mha(self.Q, self.K, self.V)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))

    def test_invalid_d_model_raises_error(self):
        """۲. بررسی صادر شدن خطای AssertionError در صورت عدم بخش‌پذیری d_model بر num_heads"""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=500, num_heads=7)

    def test_gradients_flow(self):
        """۳. بررسی انتشار صحیح گرادیان در انتشار به عقب (Backpropagation)"""
        output, _ = self.mha(self.Q, self.K, self.V)
        loss = output.sum()
        loss.backward()
        
        for name, param in self.mha.named_parameters():
            self.assertIsNotNone(param.grad, f"گرادیان برای {name} محاسبه نشد!")
            
            # این خط نوع param.grad را برای تایپ‌چکر به Tensor تغییر می‌دهد (Type Narrowing)
            assert param.grad is not None 
            
            self.assertFalse(torch.isnan(param.grad).any(), f"گرادیان {name} مقدار NaN دارد!")

    def test_3d_mask_handling(self):
        """۴. بررسی کارکرد درست ماسک سه بعدی [batch_size, seq_len, seq_len]"""
        mask = torch.tril(torch.ones(self.batch_size, self.seq_len, self.seq_len))
        output, attn_weights = self.mha(self.Q, self.K, self.V, mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))


if __name__ == "__main__":
    unittest.main()