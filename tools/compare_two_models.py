#!/usr/bin/env python3
"""Compare two GGUF models to find differences."""

import struct
from pathlib import Path
from collections import defaultdict

def analyze_gguf(path):
    info = {'tensors': [], 'metadata': {}}
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return {'error': f'Invalid magic: {magic}'}
        
        version = struct.unpack('<I', f.read(4))[0]
        num_tensors = struct.unpack('<Q', f.read(8))[0]
        num_kv = struct.unpack('<Q', f.read(8))[0]
        
        info['version'] = version
        info['num_tensors'] = num_tensors
        info['num_kv'] = num_kv
        
        # Read metadata
        for _ in range(num_kv):
            try:
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                value_type = struct.unpack('<I', f.read(4))[0]
                
                if value_type == 8:  # string
                    val_len = struct.unpack('<Q', f.read(8))[0]
                    val = f.read(val_len).decode('utf-8')
                    info['metadata'][key] = val
                elif value_type == 4:  # uint32
                    info['metadata'][key] = struct.unpack('<I', f.read(4))[0]
                elif value_type == 5:  # int32
                    info['metadata'][key] = struct.unpack('<i', f.read(4))[0]
                elif value_type == 6:  # float32
                    info['metadata'][key] = struct.unpack('<f', f.read(4))[0]
                elif value_type == 10:  # uint64
                    info['metadata'][key] = struct.unpack('<Q', f.read(8))[0]
                elif value_type == 9:  # array
                    arr_type = struct.unpack('<I', f.read(4))[0]
                    arr_len = struct.unpack('<Q', f.read(8))[0]
                    if arr_type == 4:  # uint32 array
                        arr = [struct.unpack('<I', f.read(4))[0] for _ in range(arr_len)]
                    elif arr_type == 6:  # float32 array
                        arr = [struct.unpack('<f', f.read(4))[0] for _ in range(arr_len)]
                    else:
                        arr = f'<array type {arr_type}, len {arr_len}>'
                    info['metadata'][key] = arr
                else:
                    info['metadata'][key] = f'<type {value_type}>'
            except Exception as e:
                info['metadata']['_error'] = str(e)
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
                info['tensors'].append({'name': name, 'dims': dims, 'dtype': dtype, 'offset': offset})
            except:
                break
    
    return info

# GGML types
GGML_TYPES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
    19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS',
    24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'BF16'
}

def main():
    ref_path = Path(r'C:\Users\user\ThePuppeteer\models\qwen_layered\diffusion\Qwen_Image_Layered-Q4_K_M.gguf')
    our_path = Path(r'C:\Users\user\ThePuppeteer\models\qwen_layered\diffusion\qwen_image_layered_fp8mixed-Q4_K_M.gguf')

    print('=' * 70)
    print('GGUF Model Comparison')
    print('=' * 70)

    results = {}
    for name, path in [('Reference (from internet)', ref_path), ('Our converted', our_path)]:
        print(f'\n{name}:')
        print(f'  File: {path.name}')
        print(f'  Size: {path.stat().st_size / (1024**3):.2f} GB')
        
        info = analyze_gguf(path)
        results[name] = info
        
        if 'error' in info:
            print(f'  Error: {info["error"]}')
            continue
        
        print(f'  Version: {info["version"]}')
        print(f'  Tensors: {info["num_tensors"]}')
        print(f'  KV pairs: {info["num_kv"]}')
        
        # Count tensor types
        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)
        for t in info['tensors']:
            dtype_name = GGML_TYPES.get(t['dtype'], f'unknown_{t["dtype"]}')
            type_counts[dtype_name] += 1
            # Estimate size
            numel = 1
            for d in t['dims']:
                numel *= d
            type_sizes[dtype_name] += numel
        
        print(f'\n  Tensor types:')
        for dtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f'    {dtype}: {count} tensors, ~{type_sizes[dtype]/1e9:.2f}B params')
        
        # Show some metadata
        print(f'\n  Key metadata:')
        for k in ['general.architecture', 'general.name', 'general.quantization_version']:
            if k in info['metadata']:
                print(f'    {k}: {info["metadata"][k]}')

    # Compare tensor names
    print('\n' + '=' * 70)
    print('Tensor Comparison')
    print('=' * 70)
    
    ref_info = results.get('Reference (from internet)', {})
    our_info = results.get('Our converted', {})
    
    if ref_info.get('tensors') and our_info.get('tensors'):
        ref_names = set(t['name'] for t in ref_info['tensors'])
        our_names = set(t['name'] for t in our_info['tensors'])
        
        only_in_ref = ref_names - our_names
        only_in_our = our_names - ref_names
        common = ref_names & our_names
        
        print(f'\nTensors only in reference: {len(only_in_ref)}')
        if only_in_ref:
            for name in sorted(only_in_ref)[:20]:
                print(f'  - {name}')
            if len(only_in_ref) > 20:
                print(f'  ... and {len(only_in_ref) - 20} more')
        
        print(f'\nTensors only in our model: {len(only_in_our)}')
        if only_in_our:
            for name in sorted(only_in_our)[:20]:
                print(f'  - {name}')
            if len(only_in_our) > 20:
                print(f'  ... and {len(only_in_our) - 20} more')
        
        print(f'\nCommon tensors: {len(common)}')
        
        # Check for dtype differences in common tensors
        ref_tensors = {t['name']: t for t in ref_info['tensors']}
        our_tensors = {t['name']: t for t in our_info['tensors']}
        
        dtype_diffs = []
        for name in common:
            ref_dtype = GGML_TYPES.get(ref_tensors[name]['dtype'], f'unknown_{ref_tensors[name]["dtype"]}')
            our_dtype = GGML_TYPES.get(our_tensors[name]['dtype'], f'unknown_{our_tensors[name]["dtype"]}')
            if ref_dtype != our_dtype:
                dtype_diffs.append((name, ref_dtype, our_dtype))
        
        if dtype_diffs:
            print(f'\nDtype differences in common tensors: {len(dtype_diffs)}')
            for name, ref_dt, our_dt in dtype_diffs[:20]:
                print(f'  {name}: ref={ref_dt}, our={our_dt}')
            if len(dtype_diffs) > 20:
                print(f'  ... and {len(dtype_diffs) - 20} more')

if __name__ == "__main__":
    main()
