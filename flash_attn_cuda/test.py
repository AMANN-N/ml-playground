import torch
import flash_attn_cuda

seq_len = 128
head_dim = 64
q = torch.randn(seq_len, head_dim, device='cuda')
k = torch.randn(seq_len, head_dim, device='cuda')
v = torch.randn(seq_len, head_dim, device='cuda')
out = torch.zeros_like(q)

flash_attn_cuda.flash_attn(q, k, v, out)

q_ = q.unsqueeze(0)
k_ = k.unsqueeze(0)
v_ = v.unsqueeze(0)
attn = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_)
torch.testing.assert_close(out, attn[0], atol=1e-3)

print("Test passed ✅")
