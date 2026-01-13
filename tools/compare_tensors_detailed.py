#!/usr/bin/env python3
"""Compare tensors between our GGUF and reference GGUF."""

import struct
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"

def read_gguf_tensors(path):
    """Read all tensor names from GGUF file."""
    tensors = []
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Invalid magic: {magic}")
            return tensors
        
        version = struct.unpack('<I', f.read(4))[0]
        num_tensors = struct.unpack('<Q', f.read(8))[0]
        num_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Version: {version}, Tensors: {num_tensors}, KV pairs: {num_kv}")
        
        # Skip KV pairs
        for _ in range(num_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Skip value based on type
            if value_type == 8:  # string
                val_len = struct.unpack('<Q', f.read(8))[0]
                f.read(val_len)
            elif value_type == 4:  # uint32
                f.read(4)
            elif value_type == 6:  # float32
                f.read(4)
            elif value_type == 10:  # uint64
                f.read(8)
            elif value_type == 5:  # int32
                f.read(4)
            elif value_type == 7:  # bool
                f.read(1)
            else:
                print(f"Unknown KV type: {value_type} for key: {key}")
                break
        
        # Read tensor info
        for _ in range(num_tensors):
            try:
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8')
                ndims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                tensors.append({
                    'name': name,
                    'dims': dims,
                    'dtype': dtype,
                    'offset': offset
                })
            except Exception as e:
                print(f"Error reading tensor: {e}")
                break
    
    return tensors

def main():
    our_model = DOWNLOADS / "LongCatVideo-Q4_K_M.gguf"
    ref_model = DOWNLOADS / "LongCat-Avatar-Multi_comfy-Q4_K_M.gguf"
    
    print("=" * 70)
    print("Reading OUR model tensors:")
    print("=" * 70)
    our_tensors = read_gguf_tensors(our_model)
    our_names = set(t['name'] for t in our_tensors)
    
    print(f"\nTotal tensors in our model: {len(our_tensors)}")
    
    print("\n" + "=" * 70)
    print("Reading REFERENCE model tensors:")
    print("=" * 70)
    ref_tensors = read_gguf_tensors(ref_model)
    ref_names = set(t['name'] for t in ref_tensors)
    
    print(f"\nTotal tensors in reference model: {len(ref_tensors)}")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    
    only_in_ref = ref_names - our_names
    only_in_ours = our_names - ref_names
    common = our_names & ref_names
    
    print(f"\nTensors only in REFERENCE ({len(only_in_ref)}):")
    
    # Group by prefix
    prefixes = {}
    for name in sorted(only_in_ref):
        prefix = name.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(name)
    
    for prefix, names in sorted(prefixes.items()):
        print(f"\n  {prefix}: {len(names)} tensors")
        for name in names[:5]:
            print(f"    - {name}")
        if len(names) > 5:
            print(f"    ... and {len(names) - 5} more")
    
    print(f"\n\nTensors only in OUR model ({len(only_in_ours)}):")
    for name in sorted(only_in_ours)[:20]:
        print(f"  - {name}")
    
    print(f"\n\nCommon tensors: {len(common)}")
    
    # Check if reference has VAE, text encoder, etc.
    print("\n" + "=" * 70)
    print("Reference model components:")
    print("=" * 70)
    
    components = {}
    for name in ref_names:
        parts = name.split('.')
        if len(parts) > 0:
            comp = parts[0]
            components[comp] = components.get(comp, 0) + 1
    
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {count} tensors")

if __name__ == "__main__":
    main()
