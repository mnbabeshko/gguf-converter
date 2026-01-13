#!/usr/bin/env python3
"""Compare two GGUF models to find differences."""

import struct
import os
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"

def read_gguf_header(path):
    """Read GGUF file header and tensor info."""
    info = {
        'path': path,
        'size_bytes': path.stat().st_size,
        'size_gb': path.stat().st_size / (1024**3),
    }
    
    with open(path, 'rb') as f:
        # Magic number
        magic = f.read(4)
        info['magic'] = magic.hex()
        info['magic_str'] = magic.decode('latin-1') if magic == b'GGUF' else 'INVALID'
        
        if magic != b'GGUF':
            # Try our custom format
            f.seek(0)
            custom_magic = struct.unpack('<I', f.read(4))[0]
            info['custom_magic'] = hex(custom_magic)
            version = struct.unpack('<I', f.read(4))[0]
            info['version'] = version
            num_tensors = struct.unpack('<Q', f.read(8))[0]
            info['num_tensors'] = num_tensors
            num_kv = struct.unpack('<Q', f.read(8))[0]
            info['num_kv'] = num_kv
            info['format'] = 'custom'
            
            # Read first few tensor names
            tensor_names = []
            for i in range(min(10, num_tensors)):
                try:
                    name_len = struct.unpack('<Q', f.read(8))[0]
                    name = f.read(name_len).decode('utf-8')
                    tensor_names.append(name)
                    # Skip rest of tensor header
                    ndims = struct.unpack('<I', f.read(4))[0]
                    for _ in range(ndims):
                        f.read(8)  # dimension
                    f.read(4)  # dtype
                    f.read(8)  # offset
                except:
                    break
            info['sample_tensors'] = tensor_names
        else:
            # Standard GGUF format
            version = struct.unpack('<I', f.read(4))[0]
            info['version'] = version
            num_tensors = struct.unpack('<Q', f.read(8))[0]
            info['num_tensors'] = num_tensors
            num_kv = struct.unpack('<Q', f.read(8))[0]
            info['num_kv'] = num_kv
            info['format'] = 'gguf_standard'
            
            # Try to read some metadata
            metadata = {}
            for _ in range(min(20, num_kv)):
                try:
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    key = f.read(key_len).decode('utf-8')
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    # Read value based on type
                    if value_type == 8:  # string
                        val_len = struct.unpack('<Q', f.read(8))[0]
                        val = f.read(val_len).decode('utf-8')
                        metadata[key] = val
                    elif value_type == 4:  # uint32
                        metadata[key] = struct.unpack('<I', f.read(4))[0]
                    elif value_type == 6:  # float32
                        metadata[key] = struct.unpack('<f', f.read(4))[0]
                    elif value_type == 10:  # uint64
                        metadata[key] = struct.unpack('<Q', f.read(8))[0]
                    else:
                        metadata[key] = f"<type {value_type}>"
                        break  # Stop if we hit unknown type
                except Exception as e:
                    break
            info['metadata'] = metadata
    
    return info

def main():
    our_model = DOWNLOADS / "LongCatVideo-Q4_K_M.gguf"
    ref_model = DOWNLOADS / "LongCat-Avatar-Multi_comfy-Q4_K_M.gguf"
    
    print("=" * 60)
    print("GGUF Model Comparison")
    print("=" * 60)
    
    for model_path in [our_model, ref_model]:
        if not model_path.exists():
            print(f"\n❌ File not found: {model_path.name}")
            continue
            
        print(f"\n📁 {model_path.name}")
        print("-" * 50)
        
        info = read_gguf_header(model_path)
        
        print(f"  Size: {info['size_gb']:.2f} GB ({info['size_bytes']:,} bytes)")
        print(f"  Format: {info.get('format', 'unknown')}")
        print(f"  Magic: {info.get('magic_str', info.get('custom_magic', 'unknown'))}")
        print(f"  Version: {info.get('version', 'unknown')}")
        print(f"  Num tensors: {info.get('num_tensors', 'unknown')}")
        print(f"  Num KV pairs: {info.get('num_kv', 'unknown')}")
        
        if 'metadata' in info and info['metadata']:
            print(f"\n  Metadata (first entries):")
            for k, v in list(info['metadata'].items())[:10]:
                print(f"    {k}: {v}")
        
        if 'sample_tensors' in info:
            print(f"\n  Sample tensor names:")
            for name in info['sample_tensors']:
                print(f"    - {name}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    
    if our_model.exists() and ref_model.exists():
        our_info = read_gguf_header(our_model)
        ref_info = read_gguf_header(ref_model)
        
        size_ratio = ref_info['size_bytes'] / our_info['size_bytes']
        print(f"\nReference model is {size_ratio:.2f}x larger than our model")
        
        if our_info.get('format') != ref_info.get('format'):
            print(f"\n⚠️  Different formats detected!")
            print(f"   Our model: {our_info.get('format')}")
            print(f"   Reference: {ref_info.get('format')}")

if __name__ == "__main__":
    main()
