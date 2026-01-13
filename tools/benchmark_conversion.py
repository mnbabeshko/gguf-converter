"""
Benchmark script to measure conversion speed.
Run this to compare performance before/after optimizations.
"""
import sys
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_quantization():
    """Benchmark quantization functions."""
    from gguf_converter import GGUFConverter
    
    converter = GGUFConverter()
    
    # Test different tensor sizes
    sizes = [
        (256,),           # Small
        (1024,),          # Medium 1D
        (4096, 4096),     # Large 2D (16M params)
        (8192, 8192),     # Very large 2D (67M params)
    ]
    
    qtypes = ["Q8_0", "Q4_K", "Q6_K", "Q5_K", "Q3_K"]
    
    print("=" * 60)
    print("Quantization Benchmark (Single-threaded)")
    print("=" * 60)
    
    for size in sizes:
        numel = np.prod(size)
        arr = np.random.randn(*size).astype(np.float32)
        
        print(f"\nTensor shape: {size} ({numel:,} elements)")
        print("-" * 40)
        
        for qtype in qtypes:
            times = []
            for _ in range(3):  # 3 runs for average
                start = time.perf_counter()
                try:
                    converter._quantize(arr, qtype)
                except Exception as e:
                    print(f"  {qtype}: ERROR - {e}")
                    break
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            if times:
                avg_time = sum(times) / len(times)
                throughput = numel / avg_time / 1e6  # M elements/sec
                print(f"  {qtype}: {avg_time*1000:.1f}ms ({throughput:.1f}M elem/s)")

def benchmark_parallel_quantization():
    """Benchmark parallel quantization."""
    from gguf_converter import GGUFConverter
    
    converter = GGUFConverter()
    num_workers = os.cpu_count() // 2
    
    print("\n" + "=" * 60)
    print(f"Parallel Quantization Benchmark ({num_workers} workers)")
    print("=" * 60)
    
    # Simulate batch of tensors
    num_tensors = 100
    tensor_size = (4096, 4096)
    numel = np.prod(tensor_size)
    qtype = "Q4_K"
    
    print(f"\nProcessing {num_tensors} tensors of shape {tensor_size}")
    print(f"Total elements: {num_tensors * numel:,}")
    print("-" * 40)
    
    # Generate test data
    tensors = [np.random.randn(*tensor_size).astype(np.float32) for _ in range(num_tensors)]
    
    def quantize_single(arr):
        return converter._quantize(arr, qtype)
    
    # Single-threaded
    start = time.perf_counter()
    for arr in tensors:
        quantize_single(arr)
    single_time = time.perf_counter() - start
    print(f"Single-threaded: {single_time:.1f}s ({num_tensors * numel / single_time / 1e6:.1f}M elem/s)")
    
    # Multi-threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(quantize_single, arr) for arr in tensors]
        for future in as_completed(futures):
            future.result()
    multi_time = time.perf_counter() - start
    print(f"Multi-threaded:  {multi_time:.1f}s ({num_tensors * numel / multi_time / 1e6:.1f}M elem/s)")
    print(f"Speedup: {single_time / multi_time:.2f}x")

def benchmark_numpy_conversion():
    """Benchmark numpy conversion patterns."""
    import torch
    
    print("\n" + "=" * 60)
    print("NumPy Conversion Benchmark")
    print("=" * 60)
    
    sizes = [(4096, 4096), (8192, 8192)]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for size in sizes:
        print(f"\nTensor shape: {size}")
        print("-" * 40)
        
        for dtype in dtypes:
            tensor = torch.randn(*size, dtype=dtype)
            
            # Method 1: Always convert to float
            times1 = []
            for _ in range(5):
                start = time.perf_counter()
                arr = tensor.float().numpy()
                times1.append(time.perf_counter() - start)
            
            # Method 2: Check dtype first
            times2 = []
            for _ in range(5):
                start = time.perf_counter()
                if tensor.dtype == torch.float32:
                    arr = tensor.numpy()
                else:
                    arr = tensor.float().numpy()
                times2.append(time.perf_counter() - start)
            
            avg1 = sum(times1) / len(times1) * 1000
            avg2 = sum(times2) / len(times2) * 1000
            speedup = avg1 / avg2 if avg2 > 0 else 0
            
            print(f"  {str(dtype):15s}: always_float={avg1:.1f}ms, check_first={avg2:.1f}ms, speedup={speedup:.2f}x")

if __name__ == "__main__":
    print("GGUF Converter Performance Benchmark")
    print("=" * 60)
    print(f"CPU cores: {os.cpu_count()}")
    
    benchmark_numpy_conversion()
    benchmark_quantization()
    benchmark_parallel_quantization()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


def benchmark_process_pool():
    """Benchmark ProcessPoolExecutor vs ThreadPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    # Need to use spawn for Windows compatibility
    mp.set_start_method('spawn', force=True)
    
    num_workers = os.cpu_count() // 2
    
    print("\n" + "=" * 60)
    print(f"Process Pool Benchmark ({num_workers} workers)")
    print("=" * 60)
    
    # Smaller test for process pool (serialization overhead)
    num_tensors = 20
    tensor_size = (2048, 2048)
    numel = np.prod(tensor_size)
    
    print(f"\nProcessing {num_tensors} tensors of shape {tensor_size}")
    print("-" * 40)
    
    # Generate test data
    tensors = [np.random.randn(*tensor_size).astype(np.float32) for _ in range(num_tensors)]
    
    # Single-threaded baseline
    from gguf_converter import GGUFConverter
    converter = GGUFConverter()
    
    start = time.perf_counter()
    for arr in tensors:
        converter._quantize(arr, "Q4_K")
    single_time = time.perf_counter() - start
    print(f"Single-threaded: {single_time:.2f}s")
    
    # ThreadPoolExecutor
    def quantize_thread(arr):
        return converter._quantize(arr, "Q4_K")
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(quantize_thread, tensors))
    thread_time = time.perf_counter() - start
    print(f"ThreadPool:      {thread_time:.2f}s (speedup: {single_time/thread_time:.2f}x)")
    
    print("\nNote: ProcessPoolExecutor not tested due to pickle overhead for large arrays")
