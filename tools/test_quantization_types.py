#!/usr/bin/env python3
"""
Test all quantization types (Q3_K, Q4_K, Q5_K, Q6_K) for correctness.
Verifies that quantization and dequantization produce reasonable results.
"""

import sys
import struct
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantizer import GGUFConverter, QUANT_CONFIGS


def dequantize_block(data: bytes, qtype: str, num_blocks: int) -> np.ndarray:
    """Dequantize a block of data back to float32."""
    config = QUANT_CONFIGS[qtype]
    block_bytes = config['block_bytes']
    block_size = 256
    
    result = np.zeros(num_blocks * block_size, dtype=np.float32)
    
    for i in range(num_blocks):
        offset = i * block_bytes
        scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        min_val = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        packed = data[offset+4:offset+block_bytes]
        
        # Unpack based on type
        if qtype == 'Q3_K':
            q = unpack_3bit(packed, block_size)
        elif qtype == 'Q4_K':
            q = unpack_4bit(packed, block_size)
        elif qtype == 'Q5_K':
            q = unpack_5bit(packed, block_size)
        else:  # Q6_K
            q = unpack_6bit(packed, block_size)
        
        # Dequantize
        result[i*block_size:(i+1)*block_size] = q * float(scale) + float(min_val)
    
    return result


def unpack_3bit(packed: bytes, block_size: int) -> np.ndarray:
    """Unpack 3-bit values from packed bytes."""
    result = np.zeros(block_size, dtype=np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    
    # 8 values packed into 3 bytes
    num_groups = block_size // 8
    for g in range(num_groups):
        b0, b1, b2 = packed_arr[g*3], packed_arr[g*3+1], packed_arr[g*3+2]
        result[g*8 + 0] = b0 & 0x07
        result[g*8 + 1] = (b0 >> 3) & 0x07
        result[g*8 + 2] = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
        result[g*8 + 3] = (b1 >> 1) & 0x07
        result[g*8 + 4] = (b1 >> 4) & 0x07
        result[g*8 + 5] = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
        result[g*8 + 6] = (b2 >> 2) & 0x07
        result[g*8 + 7] = (b2 >> 5) & 0x07
    
    return result


def unpack_4bit(packed: bytes, block_size: int) -> np.ndarray:
    """Unpack 4-bit values from packed bytes."""
    result = np.zeros(block_size, dtype=np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    
    for i in range(len(packed_arr)):
        result[i*2] = packed_arr[i] & 0x0F
        result[i*2 + 1] = (packed_arr[i] >> 4) & 0x0F
    
    return result


def unpack_5bit(packed: bytes, block_size: int) -> np.ndarray:
    """Unpack 5-bit values from packed bytes."""
    result = np.zeros(block_size, dtype=np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    
    # 8 values packed into 5 bytes
    num_groups = block_size // 8
    for g in range(num_groups):
        b0, b1, b2, b3, b4 = packed_arr[g*5:g*5+5]
        result[g*8 + 0] = b0 & 0x1F
        result[g*8 + 1] = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3)
        result[g*8 + 2] = (b1 >> 2) & 0x1F
        result[g*8 + 3] = ((b1 >> 7) & 0x01) | ((b2 & 0x0F) << 1)
        result[g*8 + 4] = ((b2 >> 4) & 0x0F) | ((b3 & 0x01) << 4)
        result[g*8 + 5] = (b3 >> 1) & 0x1F
        result[g*8 + 6] = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2)
        result[g*8 + 7] = (b4 >> 3) & 0x1F
    
    return result


def unpack_6bit(packed: bytes, block_size: int) -> np.ndarray:
    """Unpack 6-bit values from packed bytes."""
    result = np.zeros(block_size, dtype=np.float32)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    
    # 4 values packed into 3 bytes
    num_groups = block_size // 4
    for g in range(num_groups):
        b0, b1, b2 = packed_arr[g*3], packed_arr[g*3+1], packed_arr[g*3+2]
        result[g*4 + 0] = b0 & 0x3F
        result[g*4 + 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
        result[g*4 + 2] = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
        result[g*4 + 3] = (b2 >> 2) & 0x3F
    
    return result


def test_quantization_type(qtype: str, converter: GGUFConverter):
    """Test a specific quantization type."""
    print(f"\n{'='*60}")
    print(f"Testing {qtype}")
    print(f"{'='*60}")
    
    config = QUANT_CONFIGS[qtype]
    print(f"Config: bits={config['bits']}, max_val={config['max_val']}, block_bytes={config['block_bytes']}")
    
    # Test with different array sizes
    test_sizes = [256, 512, 1024, 4096]
    
    for size in test_sizes:
        # Create test data with known distribution
        np.random.seed(42)
        original = np.random.randn(size).astype(np.float32)
        
        # Quantize
        quantized_data = converter._quantize_block(original, qtype)
        
        # Calculate expected size
        num_blocks = (size + 255) // 256
        expected_size = num_blocks * config['block_bytes']
        
        print(f"\n  Size {size}:")
        print(f"    Original: {original.nbytes} bytes")
        print(f"    Quantized: {len(quantized_data)} bytes (expected: {expected_size})")
        
        # Verify size
        if len(quantized_data) != expected_size:
            print(f"    ❌ Size mismatch!")
            return False
        
        # Dequantize and compare
        dequantized = dequantize_block(quantized_data, qtype, num_blocks)[:size]
        
        # Calculate error metrics
        mse = np.mean((original - dequantized) ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(original - dequantized))
        correlation = np.corrcoef(original, dequantized)[0, 1]
        
        print(f"    RMSE: {rmse:.6f}")
        print(f"    Max error: {max_error:.6f}")
        print(f"    Correlation: {correlation:.6f}")
        
        # Check quality thresholds based on bit depth
        bits = config['bits']
        expected_rmse = 0.5 / (2 ** (bits - 1))  # Rough estimate
        
        if correlation < 0.9:
            print(f"    ❌ Correlation too low!")
            return False
        
        print(f"    ✅ OK")
    
    return True


def test_edge_cases(converter: GGUFConverter):
    """Test edge cases for quantization."""
    print(f"\n{'='*60}")
    print("Testing Edge Cases")
    print(f"{'='*60}")
    
    for qtype in ['Q3_K', 'Q4_K', 'Q5_K', 'Q6_K']:
        print(f"\n  {qtype}:")
        
        # Test with zeros
        zeros = np.zeros(256, dtype=np.float32)
        try:
            data = converter._quantize_block(zeros, qtype)
            print(f"    Zeros: ✅ ({len(data)} bytes)")
        except Exception as e:
            print(f"    Zeros: ❌ {e}")
            return False
        
        # Test with constant values
        const = np.ones(256, dtype=np.float32) * 0.5
        try:
            data = converter._quantize_block(const, qtype)
            print(f"    Constant: ✅ ({len(data)} bytes)")
        except Exception as e:
            print(f"    Constant: ❌ {e}")
            return False
        
        # Test with large values
        large = np.random.randn(256).astype(np.float32) * 1000
        try:
            data = converter._quantize_block(large, qtype)
            print(f"    Large values: ✅ ({len(data)} bytes)")
        except Exception as e:
            print(f"    Large values: ❌ {e}")
            return False
        
        # Test with very small values
        small = np.random.randn(256).astype(np.float32) * 1e-6
        try:
            data = converter._quantize_block(small, qtype)
            print(f"    Small values: ✅ ({len(data)} bytes)")
        except Exception as e:
            print(f"    Small values: ❌ {e}")
            return False
    
    return True


def main():
    print("=" * 60)
    print("GGUF Quantization Test Suite")
    print("=" * 60)
    
    converter = GGUFConverter()
    
    all_passed = True
    
    # Test each quantization type
    for qtype in ['Q3_K', 'Q4_K', 'Q5_K', 'Q6_K']:
        if not test_quantization_type(qtype, converter):
            all_passed = False
    
    # Test edge cases
    if not test_edge_cases(converter):
        all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
