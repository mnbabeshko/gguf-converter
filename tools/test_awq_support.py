"""
Test AWQ Support for GGUF Converter

Tests:
1. AWQ format detection
2. INT4 unpacking correctness
3. Dequantization accuracy
4. End-to-end AWQ -> GGUF conversion

Usage:
    python tools/test_awq_support.py
    python tools/test_awq_support.py --model path/to/awq_model.safetensors
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_int4_unpacking():
    """Test INT4 unpacking from INT32."""
    print("\n=== Test 1: INT4 Unpacking ===")
    
    from awq_support import AWQDequantizer
    
    # Create test data: pack known INT4 values into INT32
    # Values 0-7 packed into first int32, 8-15 into second
    test_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int8)
    
    # Pack into INT32 (8 values per int32) - use unsigned to avoid overflow
    packed = np.zeros(2, dtype=np.uint32)
    for i in range(8):
        packed[0] |= np.uint32(test_values[i]) << (4 * i)
        packed[1] |= np.uint32(test_values[8 + i]) << (4 * i)
    
    # Convert to int32 for the function
    packed = packed.astype(np.int32).reshape(1, 2)  # [1, 2] shape
    
    # Unpack
    unpacked = AWQDequantizer.unpack_int4_fast(packed)
    
    # Verify
    expected = test_values.reshape(1, 16)
    
    print(f"  Packed shape: {packed.shape}")
    print(f"  Unpacked shape: {unpacked.shape}")
    print(f"  Expected: {expected.flatten()}")
    print(f"  Got:      {unpacked.flatten()}")
    
    if np.array_equal(unpacked, expected):
        print("  ✓ INT4 unpacking PASSED")
        return True
    else:
        print("  ✗ INT4 unpacking FAILED")
        return False


def test_dequantization():
    """Test AWQ dequantization formula."""
    print("\n=== Test 2: Dequantization ===")
    
    from awq_support import AWQDequantizer
    
    dequant = AWQDequantizer()
    
    # Create synthetic AWQ tensors
    # qweight: [4, 2] -> unpacks to [4, 16]
    # qzeros: [1, 2] -> unpacks to [1, 16] (1 group)
    # scales: [1, 16]
    # group_size: 4 (4 output features per group)
    
    # Pack some test values - use unsigned to avoid overflow
    qweight_packed = np.zeros((4, 2), dtype=np.uint32)
    qzeros_packed = np.zeros((1, 2), dtype=np.uint32)
    
    # Fill with simple pattern: all 8s for weights, all 0s for zeros
    for i in range(8):
        qweight_packed[:, 0] |= np.uint32(8) << (4 * i)
        qweight_packed[:, 1] |= np.uint32(8) << (4 * i)
        # zeros stay 0
    
    # Convert to int32
    qweight_packed = qweight_packed.astype(np.int32)
    qzeros_packed = qzeros_packed.astype(np.int32)
    
    # Scales: all 0.5
    scales = np.full((1, 16), 0.5, dtype=np.float16)
    
    # Dequantize
    result = dequant.dequantize_layer_numpy(qweight_packed, qzeros_packed, scales, group_size=4)
    
    # Expected: (8 - 0) * 0.5 = 4.0 for all values
    expected = np.full((4, 16), 4.0, dtype=np.float16)
    
    print(f"  qweight shape: {qweight_packed.shape}")
    print(f"  Result shape: {result.shape}")
    print(f"  Expected value: 4.0")
    print(f"  Got values: min={result.min():.2f}, max={result.max():.2f}, mean={result.mean():.2f}")
    
    if np.allclose(result, expected, rtol=1e-2):
        print("  ✓ Dequantization PASSED")
        return True
    else:
        print("  ✗ Dequantization FAILED")
        return False


def test_awq_detection_negative():
    """Test that non-AWQ models are not detected as AWQ."""
    print("\n=== Test 3: AWQ Detection (Negative) ===")
    
    from quantizer import detect_awq_model
    
    # Create a temporary non-AWQ safetensors file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Create minimal safetensors with regular tensors
        from safetensors.torch import save_file
        
        tensors = {
            "model.layer.weight": torch.randn(64, 64),
            "model.layer.bias": torch.randn(64),
        }
        save_file(tensors, temp_path)
        
        # Test detection
        result = detect_awq_model(temp_path)
        
        print(f"  is_awq: {result.is_awq}")
        
        if not result.is_awq:
            print("  ✓ Non-AWQ detection PASSED")
            return True
        else:
            print("  ✗ Non-AWQ detection FAILED (false positive)")
            return False
    finally:
        temp_path.unlink()


def test_awq_detection_positive():
    """Test that AWQ models are correctly detected."""
    print("\n=== Test 4: AWQ Detection (Positive) ===")
    
    from quantizer import detect_awq_model
    
    # Create a temporary AWQ-like safetensors file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        from safetensors.torch import save_file
        
        # Create AWQ-style tensors
        out_features = 256
        in_features = 256
        group_size = 128
        num_groups = out_features // group_size
        
        tensors = {
            "model.layer.qweight": torch.randint(0, 2**31, (out_features, in_features // 8), dtype=torch.int32),
            "model.layer.qzeros": torch.randint(0, 2**31, (num_groups, in_features // 8), dtype=torch.int32),
            "model.layer.scales": torch.randn(num_groups, in_features, dtype=torch.float16),
        }
        save_file(tensors, temp_path)
        
        # Test detection
        result = detect_awq_model(temp_path)
        
        print(f"  is_awq: {result.is_awq}")
        print(f"  num_layers: {result.num_layers}")
        print(f"  group_size: {result.group_size}")
        print(f"  bits: {result.bits}")
        
        if result.is_awq and result.num_layers == 1:
            print("  ✓ AWQ detection PASSED")
            return True
        else:
            print("  ✗ AWQ detection FAILED")
            return False
    finally:
        temp_path.unlink()


def test_real_awq_model(model_path: str):
    """Test with a real AWQ model file."""
    print(f"\n=== Test 5: Real AWQ Model ===")
    print(f"  Model: {model_path}")
    
    from quantizer import detect_awq_model, ModelAnalyzer
    from awq_support import AWQDequantizer
    
    path = Path(model_path)
    if not path.exists():
        print(f"  ✗ Model file not found: {model_path}")
        return False
    
    # Detect AWQ
    awq_info = detect_awq_model(path)
    print(f"  is_awq: {awq_info.is_awq}")
    
    if not awq_info.is_awq:
        print("  ✗ Not an AWQ model")
        return False
    
    print(f"  num_layers: {awq_info.num_layers}")
    print(f"  group_size: {awq_info.group_size}")
    print(f"  bits: {awq_info.bits}")
    
    # Analyze model
    info = ModelAnalyzer.analyze(path)
    print(f"  model_name: {info.model_name}")
    print(f"  size: {info.size_bytes / 1024 / 1024:.1f} MB")
    
    # Estimate memory
    dequant = AWQDequantizer()
    mem = dequant.estimate_memory_usage(path)
    print(f"  AWQ size: {mem['awq_size_mb']:.0f} MB")
    print(f"  FP16 size: {mem['fp16_size_mb']:.0f} MB")
    print(f"  Peak memory: {mem['peak_memory_mb']:.0f} MB")
    
    print("  ✓ Real AWQ model analysis PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test AWQ support')
    parser.add_argument('--model', type=str, help='Path to AWQ model for testing')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AWQ Support Tests")
    print("=" * 60)
    
    results = []
    
    # Basic tests
    results.append(("INT4 Unpacking", test_int4_unpacking()))
    results.append(("Dequantization", test_dequantization()))
    results.append(("AWQ Detection (Negative)", test_awq_detection_negative()))
    results.append(("AWQ Detection (Positive)", test_awq_detection_positive()))
    
    # Real model test if provided
    if args.model:
        results.append(("Real AWQ Model", test_real_awq_model(args.model)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
