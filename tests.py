import torch
from triton_attention import *
import argparse
import gc


def gpu_cleanup():
    """
    Function to clean up GPU memory.
    """
    gc.collect()
    torch.cuda.empty_cache()


def test_create_tensors(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    """
    Create tensors for attention computation.
    """
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    return q, k, v


def test_reference_computation(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    """
    Perform reference attention computation on the GPU.
    """
    q, k, v = test_create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)

    sm_scale = 0.5
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda", dtype=dtype))
    p_gpu = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        p_gpu = torch.where(M == 0, float("-inf"), p_gpu)

    p_gpu = torch.softmax(p_gpu, dim=-1)
    ref_c_gpu = torch.sum(p_gpu, dim=2)
    ref_out_gpu = torch.matmul(p_gpu, v)
    return ref_out_gpu, ref_c_gpu


def test_triton_computation(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    """
    Perform Triton-based attention computation on the GPU.
    """
    q, k, v = test_create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    sm_scale = 0.5
    torch.cuda.synchronize()
    tri_out, tri_c, tri_m = attention(q, k, v, causal, sm_scale)
    torch.cuda.synchronize()
    return tri_out, tri_c, tri_m


def test_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    gpu_cleanup()
    ref_out_gpu, ref_c_gpu = test_reference_computation(Z, H, N_CTX, HEAD_DIM, causal, dtype)
    tri_out_gpu, tri_c_gpu, tri_m_gpu = test_triton_computation(Z, H, N_CTX, HEAD_DIM, causal, dtype)

    # Convert reference tensors to match dtype of Triton results
    ref_c_gpu = ref_c_gpu.half()
    ref_out_gpu = ref_out_gpu.half()

    # Compare results
    assert torch.allclose(ref_out_gpu, tri_out_gpu.half(), atol=1e-2, rtol=0), "Attention output mismatch"
    print("Attention check passed")
    assert torch.allclose(ref_c_gpu, tri_c_gpu.half(), atol=1e-2, rtol=0), "Attention score acc mismatch"
    print("Attention score acc check passed")


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
    test_attention(args.Z, args.H, args.N_CTX, args.HEAD_DIM, args.causal)