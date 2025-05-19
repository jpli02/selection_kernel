import torch
import torch.nn as nn
# from selection_kernel import selection_attention
import math
import pandas as pd

def gpu_cleanup():
    """
    Function to clean up GPU memory.
    """
    torch.cuda.empty_cache()

def create_tensors(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    """
    Create tensors for attention computation.
    """
    # torch.manual_seed(20)
    q = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.rand((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    return q, k, v


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

def ref_attention(query_states, key_states, value_states, causal=True):
    bsz, H, q_len, d_model = query_states.shape
    attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(d_model)
                
    attention_mask = _make_causal_mask(
        bsz=bsz,
        tgt_len=q_len,
        past_key_values_length=0,
        dtype=query_states.dtype,
        device=query_states.device,
    )
        
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    cumulative_attn_map = attn_weights.sum(2)
    return attn_output, cumulative_attn_map

def triton_attention(q, k, v, causal=True):
    """
    Perform Triton-based attention computation on the GPU.
    """
    sm_scale = 1.0 / math.sqrt(128)
    tri_out, tri_c, tri_m = selection_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), causal, sm_scale)
    tri_c = tri_c.permute(0, 2, 1)
    return tri_out, tri_c, tri_m

def test_attention():
    """
    Test to compare correctness of triton cmul attention kernel
    """
    # gpu_cleanup()
    q, k, v = create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    
    Z, H, N_CTX, dim = q.shape  # Assuming q.shape is (batch_size, head_size, context_length, dim)
    print(f"batch: {Z}, Head: {H}, seqLen: {N_CTX}, dim: {dim}")

    # Get reference output from standard attention
    ref_out_gpu, ref_c_gpu = ref_attention(q, k, v)
    
    # Ensure ref output is in the correct dtype
    ref_out_gpu = ref_out_gpu.half()
    ref_c_gpu = ref_c_gpu.half()

    # Get Triton-based output
    tri_out_gpu, tri_c_gpu, tri_m_gpu = triton_attention(q, k, v)
    
    print(f"Attention max diff: {(tri_out_gpu - ref_out_gpu).abs().max().item()}")
    assert torch.allclose(ref_out_gpu, tri_out_gpu, atol=0.08, rtol=0), "Attention output mismatch"
    print("Attention check passed")
    
    print(f"Accum score max diff: {(tri_c_gpu - ref_c_gpu).abs().max().item()}")
    assert torch.allclose(ref_c_gpu, tri_c_gpu.half(), atol=0.05, rtol=0), "Attention score accumulation mismatch"
    print("Attention score accumulation check passed")
    
    # df = pd.DataFrame({
    #     'Reference Scores': ref_out_gpu.cpu().numpy().flatten(),
    #     'Triton Scores': tri_out_gpu.cpu().numpy().flatten()
    # })

    # Save to CSV with two columns
    # df.to_csv("comparison_scores.csv", index=False, float_format="%.5f")

if __name__ == "__main__":
    # Run the attention test
    test_attention()
