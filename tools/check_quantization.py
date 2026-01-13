#!/usr/bin/env python3
"""Check if our quantization is correct."""

import struct
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"

def analyze_gguf_quantization(path):
    """Analyze quantization in GGUF file."""
    print(f"\n📁 {path.name}")
    print(f"   File size: {path.stat().st_size / (1024**3):.2f} GB")
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        num_tensors = struct.unpack('<Q', f.read(8))[0]
        num_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"   Tensors: {num_tensors}, KV pairs: {num_kv}")
        
        # Skip KV pairs
        for _ in range(num_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            value_type = struct.unpack('<I', f.read(4))[0]
            
            if value_type == 8:  # string
                val_len = struct.unpack('<Q', f.read(8))[0]
                f.read(val_len)
            elif value_type in [4, 5, 6]:  # uint32, int32, float32
                f.read(4)
            elif value_type == 10:  # uint64
                f.read(8)
            elif value_type == 7:  # bool
                f.read(1)
        
        # Read tensor info and analyze dtypes
        dtypes = {}
        tensor_info = []
        for _ in range(num_tensors):
            try:
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8')
                ndims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                
                # Calculate expected size
                num_elements = 1
                for d in dims:
                    num_elements *= d
                
                tensor_info.append({
                    'name': name,
                    'dims': dims,
                    'dtype': dtype,
                    'elements': num_elements
                })
            except:
                break
        
        # GGUF dtype codes
        dtype_names = {
            0: 'F32',
            1: 'F16',
            2: 'Q4_0',
            3: 'Q4_1',
            6: 'Q5_0',
            7: 'Q5_1',
            8: 'Q8_0',
            9: 'Q8_1',
            10: 'Q2_K',
            11: 'Q3_K',
            12: 'Q4_K',
            13: 'Q5_K',
            14: 'Q6_K',
            15: 'Q8_K',
            16: 'IQ2_XXS',
            17: 'IQ2_XS',
            18: 'IQ3_XXS',
            19: 'IQ1_S',
            20: 'IQ4_NL',
            21: 'IQ3_S',
            22: 'IQ2_S',
            23: 'IQ4_XS',
            24: 'I8',
            25: 'I16',
            26: 'I32',
            27: 'I64',
            28: 'F64',
            29: 'BF16',
        }
        
        print(f"\n   Dtype distribution:")
        for dtype, count in sorted(dtypes.items()):
            name = dtype_names.get(dtype, f'UNKNOWN({dtype})')
            print(f"     {name}: {count} tensors")
        
        # Calculate theoretical size
        total_elements = sum(t['elements'] for t in tensor_info)
        print(f"\n   Total elements: {total_elements:,}")
        
        # For Q4_K_M, each element is ~4.5 bits = 0.5625 bytes
        # For our custom Q4 (dtype 14), we use 4 bits = 0.5 bytes + scale
        theoretical_q4 = total_elements * 0.5 / (1024**3)
        theoretical_f16 = total_elements * 2 / (1024**3)
        
        print(f"   Theoretical F16 size: {theoretical_f16:.2f} GB")
        print(f"   Theoretical Q4 size: {theoretical_q4:.2f} GB")
        
        return tensor_info, dtypes

def main():
    our_model = DOWNLOADS / "LongCatVideo-Q4_K_M.gguf"
    ref_model = DOWNLOADS / "LongCat-Avatar-Multi_comfy-Q4_K_M.gguf"
    
    print("=" * 60)
    print("Quantization Analysis")
    print("=" * 60)
    
    if our_model.exists():
        our_info, our_dtypes = analyze_gguf_quantization(our_model)
    
    if ref_model.exists():
        ref_info, ref_dtypes = analyze_gguf_quantization(ref_model)
    
    # Compare
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    # Check source model
    from pathlib import Path
    import json
    
    source = Path.home() / "Downloads" / "mixamo" / "diffusion_pytorch_model.safetensors"
    if source.exists():
        with open(source, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size).decode('utf-8'))
        
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        total_params = 0
        for name, info in tensors.items():
            params = 1
            for s in info.get('shape', []):
                params *= s
            total_params += params
        
        print(f"\nSource model (BF16): {source.stat().st_size / (1024**3):.2f} GB")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Expected Q4 size: {total_params * 0.5 / (1024**3):.2f} GB")
        print(f"  Our output size: {our_model.stat().st_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
