#!/usr/bin/env python3
"""Analyze source safetensors model."""

import struct
import json
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"
MIXAMO = Path.home() / "Downloads" / "mixamo"

def analyze_safetensors(path):
    """Analyze safetensors file."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))
    
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    metadata = header.get("__metadata__", {})
    
    print(f"📁 {path.name}")
    print(f"  File size: {path.stat().st_size / (1024**3):.2f} GB")
    print(f"  Total tensors: {len(tensors)}")
    
    # Analyze tensor dtypes
    dtypes = {}
    total_params = 0
    for name, info in tensors.items():
        dtype = info.get('dtype', 'unknown')
        shape = info.get('shape', [])
        params = 1
        for s in shape:
            params *= s
        total_params += params
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Dtypes: {dtypes}")
    
    # Sample tensor names
    print(f"\n  Sample tensor names (first 20):")
    for name in list(tensors.keys())[:20]:
        info = tensors[name]
        print(f"    - {name}: {info.get('dtype')} {info.get('shape')}")
    
    # Check for patterns
    print(f"\n  Tensor name patterns:")
    prefixes = {}
    for name in tensors.keys():
        prefix = name.split('.')[0] if '.' in name else name
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"    {prefix}: {count} tensors")
    
    return tensors

def main():
    print("=" * 60)
    print("Source Model Analysis")
    print("=" * 60)
    
    # Check mixamo folder first
    mixamo_model = MIXAMO / "diffusion_pytorch_model.safetensors"
    if mixamo_model.exists():
        print(f"\n📂 Found source model in mixamo folder:")
        analyze_safetensors(mixamo_model)
    else:
        print(f"\n❌ Source model not found at: {mixamo_model}")
    
    # Also check Downloads for any other safetensors
    print("\n" + "=" * 60)
    print("Other safetensors in Downloads:")
    print("=" * 60)
    
    safetensors_files = list(DOWNLOADS.glob("*.safetensors"))
    for sf in safetensors_files:
        print()
        analyze_safetensors(sf)
        print()

if __name__ == "__main__":
    main()
