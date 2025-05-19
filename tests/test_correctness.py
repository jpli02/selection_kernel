import torch
import torch.nn as nn
import math
import pytest
from selection_kernel import select_attn_func, selection_attention



def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def ref_attention(q, k, v, bias=None, causal=False):
    """
    Reference attention implementation for testing, with optional bias + causal masking.
    """
    Z, N_CTX, H, HEAD_DIM = q.shape

    # (batch, seq_q, heads, head_dim) → (batch, heads, seq_q, head_dim)
    q = q.transpose(1, 2)  
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # compute scaled dot-product scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

    # add bias if supplied
    if bias is not None:
        # bias is expected shape (batch, heads, seq_q, seq_k)
        scores = scores + bias

    # causal mask
    if causal:
        mask = _make_causal_mask(
            bsz=Z,
            tgt_len=N_CTX,
            past_key_values_length=0,
            dtype=scores.dtype,
            device=scores.device,
        )
        scores = scores + mask

    # softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # output
    out = torch.matmul(attn_weights, v)

    # back to (batch, seq_q, heads, head_dim)
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [32, 64, 128])  # Test different head dimensions
def test_flash_attention_basic(dtype, causal, d):
    """Test basic forward pass with different configurations."""
    torch.manual_seed(42)
    batch_size, seqlen_q, seqlen_k, nheads = 2, 64, 64, 4
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, dtype=dtype, device='cuda')
    k = torch.randn(batch_size, seqlen_k, nheads, d, dtype=dtype, device='cuda')
    v = torch.randn(batch_size, seqlen_k, nheads, d, dtype=dtype, device='cuda')
    
    # Test without bias
    out_flash = select_attn_func(q, k, v, None, causal=causal)
    out_ref = ref_attention(q, k, v, causal=causal)
    
    print(f"max error:  {(out_flash - out_ref).abs().max().item()} \n") 
    assert torch.allclose(out_flash, out_ref, rtol=1e-2, atol=1e-2), "Output mismatch without bias"
    
    # Test with bias
    # bias = torch.randn(batch_size, nheads, seqlen_q, seqlen_k, dtype=dtype, device='cuda')
    # out_flash_bias = select_attn_func(q, k, v, bias=bias, causal=causal)
    # out_ref_bias = ref_attention(q, k, v, bias=bias, causal=causal)
    
    # assert torch.allclose(out_flash_bias, out_ref_bias, rtol=1e-2, atol=1e-2), "Output mismatch with bias"

if __name__ == "__main__":
    pytest.main([__file__])