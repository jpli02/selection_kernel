import torch
import torch.nn as nn
from selection_kernel import selection_attention
import argparse
import math
import gc
import pandas as pd
import time

def gpu_cleanup():
    """
    Function to clean up GPU memory.
    """
    gc.collect()
    torch.cuda.empty_cache()

def create_tensors(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    """
    Create tensors for attention computation.
    """
    # torch.manual_seed(20)
    torch.manual_seed(int(time.time()))
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

def ref_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    q, k, v = create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    attn_weights = torch.matmul(q, k.transpose(2,3)) / math.sqrt(HEAD_DIM)
    
    if causal:           
        attention_mask = _make_causal_mask(
            bsz=Z,
            tgt_len=N_CTX,
            past_key_values_length=0,
            dtype=q.dtype,
            device=q.device,
        )
            
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    cumulative_attn_map = attn_weights.sum(2)
    return attn_output, cumulative_attn_map
                
def triton_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    """
    Perform Triton-based attention computation on the GPU.
    """
    q, k, v = create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    tri_out, tri_c, tri_m = selection_attention(q, k, v, causal, sm_scale)
    return tri_out, tri_c, tri_m

def flex_attn(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    q, k, v = create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    nn.attention.flex_attention()
    return tri_out, tri_c, tri_m
    
def test_attention(Z, H, N_CTX, HEAD_DIM, causal=False, dtype=torch.float16):
    """
    Test to compare correctness of triton cmul attention kernel
    """
    gpu_cleanup()
    ref_out_gpu1, ref_c_gpu1 = ref_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype)
    # Convert reference tensors to match dtype of Triton results
    ref_c_gpu1 = ref_c_gpu1.half()
    ref_out_gpu1 = ref_out_gpu1.half()
    tri_out_gpu, tri_c_gpu, tri_m_gpu = triton_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype)
    
    # Compare results
    print(f"Attention max diff: {(tri_out_gpu.half() - ref_out_gpu1).abs().max().item()}")
    assert torch.allclose(ref_out_gpu1, tri_out_gpu.half(), atol=0.07, rtol=0), "Attention output mismatch"
    print("Attention check passed")
    
    print(f"accum score max diff: {(tri_c_gpu.half() - ref_c_gpu1).abs().max().item()}")
    assert torch.allclose(ref_c_gpu1, tri_c_gpu.half(), atol=0.05, rtol=0), "Attention score acc mismatch"
    print("Attention score acc check passed")
    
    # save results
    # pd.DataFrame(ref_c_gpu.cpu().numpy().flatten()).to_csv("/u/ndani/selection_kernel/reference_scores.csv", index=False, header=False, float_format="%.5f") 
    # pd.DataFrame(tri_c_gpu.cpu().numpy().flatten()).to_csv("/u/ndani/selection_kernel/ours_scores.csv", index=False, header=False, float_format="%.5f")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Triton-based attention implementation.")
    parser.add_argument("--Z", type=int, required=True, help="Batch size")
    parser.add_argument("--H", type=int, required=True, help="Number of heads")
    parser.add_argument("--N_CTX", type=int, required=True, help="Context length")
    parser.add_argument("--HEAD_DIM", type=int, required=True, help="Head dimension")
    parser.add_argument("--causal", action="store_true", help="Enable causal attention")  # Boolean flag
    args = parser.parse_args()

    # Print arguments for debugging
    print(f"Arguments: {args}")

    # Execute the test
    # for i in range(10):
    #     test_attention(args.Z, args.H, args.N_CTX, args.HEAD_DIM, args.causal)
    #     test_attention(2 * args.Z, args.H, args.N_CTX, args.HEAD_DIM, args.causal)
    #     test_attention(args.Z, 2 * args.H, args.N_CTX, args.HEAD_DIM, args.causal)
    
    test_attention(args.Z, args.H, args.N_CTX, args.HEAD_DIM, args.causal)
    