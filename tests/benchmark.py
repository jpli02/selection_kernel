import torch
import torch.nn.functional as F
from selection_kernel import selection_attention
import argparse
import gc
import time


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

def test_ref_torch(q, k, v):
    """
    Perform reference attention computation on the GPU.
    """
    sm_scale = 0.5
    p_gpu = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    p_gpu = torch.softmax(p_gpu, dim=-1)
    ref_c_torch = torch.sum(p_gpu, dim=2)
    ref_out_torch = torch.matmul(p_gpu, v)
    return ref_out_torch, ref_c_torch

def test_ref_fa(q, k, v):
    # use flash-attention optimization by default
    ref_out_fa = F.scaled_dot_product_attention(q, k, v)    
    return ref_out_fa
    
def test_triton_computation(q, k, v, causal=False):
    """
    Perform Triton-based attention computation on the GPU.
    """
    sm_scale = 0.5
    # torch.cuda.synchronize()
    tri_out, tri_c, tri_m = selection_attention(q, k, v, causal, sm_scale)
    # torch.cuda.synchronize()
    return tri_out, tri_c, tri_m

def test_attention(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    gpu_cleanup()
    q, k, v = test_create_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    
    for i in range(10):
        test_triton_computation(q, k, v)
        
    # test for triton implementation
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_triton = time.perf_counter()
    tri_out_gpu, tri_c_gpu, tri_m_gpu = test_triton_computation(q, k, v)
    torch.cuda.synchronize()
    end_triton = time.perf_counter()
    used_mem_triton = torch.cuda.max_memory_allocated()
    latency_triton = end_triton - start_triton
    
    # test for torch implementation
    # for i in range(10):
    #     test_ref_torch(q, k, v)
        
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()
    
    # start_torch = time.perf_counter()
    # # ref_out_torch, ref_c_torch = test_ref_torch(q, k, v)
    # torch.cuda.synchronize()
    # end_torch = time.perf_counter()
    
    # used_mem_torch = torch.cuda.max_memory_allocated()
    # latency_torch = end_torch - start_torch
    
    print(f"mem usage for triton  is {used_mem_triton/ 1024**2} MB")
    # print(f"mem usage for pytorch  is {used_mem_torch / 1024**2} MB")
    
    print(f"latency for normal triton  is {latency_triton * 1000} ms")
    
    # print(f"latency for normal pytorch  is {latency_torch * 1000} ms")
    
    for i in range(10):
        test_ref_fa(q, k, v)
        
    # test for torch fa 
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_fa = time.perf_counter()
    ref_out_fa = test_ref_fa(q, k, v)
    torch.cuda.synchronize()
    end_fa = time.perf_counter()
    
    used_mem_fa = torch.cuda.max_memory_allocated()
    latency_fa = end_fa - start_fa
    # print(f"mem usage for fa implementation is {used_mem_fa / 1024**2} MB")
    # print(f"latency for normal fa implementation is {latency_fa}")
    
    # Convert reference tensors to match dtype of Triton results
    ref_c_torch = ref_c_torch.half()
    ref_out_torch = ref_out_torch.half()

    # Compare results
    assert torch.allclose(ref_out_torch, tri_out_gpu.half(), atol=1e-2, rtol=0), "Attention output mismatch"
    print("Attention check passed")
    assert torch.allclose(ref_c_torch, tri_c_gpu.half(), atol=1e-2, rtol=0), "Attention score acc mismatch"
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