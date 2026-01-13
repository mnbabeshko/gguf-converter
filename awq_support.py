"""
AWQ Support Module for GGUF Converter
Dequantization of AWQ (Activation-aware Weight Quantization) models to FP16

AWQ Format:
- qweight: INT32 tensor with packed INT4 values (8 values per int32)
- qzeros: INT32 tensor with packed zero points
- scales: FP16 tensor with per-group scales
- group_size: typically 128 (number of weights sharing same scale/zero)

Dequantization formula:
    weight_fp16 = (qweight_int4 - qzeros_int4) * scales

Author: miha2017
"""

import struct
import gc
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Generator, Tuple
from dataclasses import dataclass

import torch
import numpy as np

# Get the module logger
file_logger = logging.getLogger('gguf_converter')


@dataclass
class AWQLayerInfo:
    """Information about a single AWQ quantized layer."""
    name: str
    qweight_shape: tuple
    scales_shape: tuple
    qzeros_shape: tuple
    group_size: int
    out_features: int
    in_features: int


class AWQDequantizer:
    """
    Dequantizer for AWQ (Activation-aware Weight Quantization) models.
    
    Converts INT4 AWQ weights back to FP16 for further processing.
    Supports streaming/layer-by-layer dequantization for memory efficiency.
    """
    
    def __init__(self, 
                 progress_cb: Optional[Callable] = None,
                 log_cb: Optional[Callable] = None,
                 low_memory: bool = False):
        """
        Initialize AWQ dequantizer.
        
        Args:
            progress_cb: Callback for progress updates (percent, status)
            log_cb: Callback for log messages
            low_memory: If True, use memory-optimized processing
        """
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.low_memory = low_memory
        self._cancelled = False
    
    def cancel(self):
        """Cancel the current operation."""
        self._cancelled = True
    
    def log(self, msg: str):
        """Log message."""
        if self.log_cb:
            self.log_cb(msg)
        file_logger.info(msg)
    
    def progress(self, val: int, status: str):
        """Update progress."""
        if self.progress_cb:
            self.progress_cb(val, status)
    
    @staticmethod
    def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack INT4 values from INT32 packed format.
        
        AWQ packs 8 INT4 values into each INT32:
        - Bits 0-3: value 0
        - Bits 4-7: value 1
        - ...
        - Bits 28-31: value 7
        
        Args:
            packed: INT32 tensor of shape [N, K // 8]
        
        Returns:
            INT8 tensor of shape [N, K] with values 0-15
        """
        # Ensure we're working with int32
        if packed.dtype != torch.int32:
            packed = packed.to(torch.int32)
        
        # Output shape: expand K dimension by 8
        out_shape = list(packed.shape)
        out_shape[-1] *= 8
        
        # Unpack 8 INT4 values from each INT32
        result = torch.zeros(out_shape, dtype=torch.int8, device=packed.device)
        
        for i in range(8):
            result[..., i::8] = ((packed >> (4 * i)) & 0xF).to(torch.int8)
        
        return result
    
    @staticmethod
    def unpack_int4_fast(packed: np.ndarray) -> np.ndarray:
        """
        Fast numpy version of INT4 unpacking.
        
        Args:
            packed: INT32 array of shape [N, K // 8]
        
        Returns:
            INT8 array of shape [N, K] with values 0-15
        """
        # Ensure int32
        if packed.dtype != np.int32:
            packed = packed.astype(np.int32)
        
        # Output shape
        out_shape = list(packed.shape)
        out_shape[-1] *= 8
        
        result = np.zeros(out_shape, dtype=np.int8)
        
        for i in range(8):
            result[..., i::8] = ((packed >> (4 * i)) & 0xF).astype(np.int8)
        
        return result
    
    def dequantize_layer(self,
                         qweight: torch.Tensor,
                         qzeros: torch.Tensor,
                         scales: torch.Tensor,
                         group_size: int = 128) -> torch.Tensor:
        """
        Dequantize a single AWQ layer to FP16.
        
        Args:
            qweight: Packed INT4 weights [out_features, in_features // 8]
            qzeros: Packed INT4 zero points [out_features // group_size, in_features // 8]
            scales: FP16 scales [out_features // group_size, in_features]
            group_size: Number of weights per group (default 128)
        
        Returns:
            FP16 weight tensor [out_features, in_features]
        """
        # Unpack INT4 values
        weight_int4 = self.unpack_int4(qweight)  # [out_features, in_features]
        zeros_int4 = self.unpack_int4(qzeros)    # [num_groups, in_features]
        
        out_features = weight_int4.shape[0]
        in_features = weight_int4.shape[1]
        num_groups = scales.shape[0]
        
        # Ensure scales is FP16
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        
        # Expand zeros and scales to match weight shape
        # zeros: [num_groups, in_features] -> [out_features, in_features]
        # scales: [num_groups, in_features] -> [out_features, in_features]
        
        # Each group of `group_size` output features shares the same zeros/scales
        zeros_expanded = zeros_int4.repeat_interleave(group_size, dim=0)[:out_features]
        scales_expanded = scales.repeat_interleave(group_size, dim=0)[:out_features]
        
        # Dequantize: weight_fp16 = (weight_int4 - zeros) * scales
        weight_fp16 = (weight_int4.to(torch.float16) - zeros_expanded.to(torch.float16)) * scales_expanded
        
        return weight_fp16
    
    def dequantize_layer_numpy(self,
                               qweight: np.ndarray,
                               qzeros: np.ndarray,
                               scales: np.ndarray,
                               group_size: int = 128) -> np.ndarray:
        """
        Numpy version of layer dequantization (for CPU-only processing).
        
        Args:
            qweight: Packed INT4 weights [out_features, in_features // 8]
            qzeros: Packed INT4 zero points [num_groups, in_features // 8]
            scales: FP16 scales [num_groups, in_features]
            group_size: Number of weights per group
        
        Returns:
            FP16 weight array [out_features, in_features]
        """
        # Unpack INT4 values
        weight_int4 = self.unpack_int4_fast(qweight)  # [out_features, in_features]
        zeros_int4 = self.unpack_int4_fast(qzeros)    # [num_groups, in_features]
        
        out_features = weight_int4.shape[0]
        in_features = weight_int4.shape[1]
        
        # Expand zeros and scales
        zeros_expanded = np.repeat(zeros_int4, group_size, axis=0)[:out_features]
        scales_expanded = np.repeat(scales, group_size, axis=0)[:out_features]
        
        # Dequantize
        weight_fp16 = (weight_int4.astype(np.float16) - zeros_expanded.astype(np.float16)) * scales_expanded.astype(np.float16)
        
        return weight_fp16

    
    def dequantize_model_streaming(self,
                                   model_path: Path,
                                   output_callback: Callable[[str, np.ndarray], None],
                                   group_size: int = 128) -> int:
        """
        Stream-dequantize AWQ model, calling callback for each dequantized tensor.
        
        This method processes tensors one at a time to minimize memory usage.
        Non-AWQ tensors are passed through unchanged.
        
        Args:
            model_path: Path to AWQ safetensors file
            output_callback: Called with (tensor_name, tensor_data) for each tensor
            group_size: AWQ group size (default 128)
        
        Returns:
            Number of dequantized layers
        """
        from safetensors import safe_open
        
        self.log(f"Starting AWQ dequantization: {model_path.name}")
        
        # First pass: collect AWQ layer info
        with safe_open(model_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
        
        # Find AWQ layers (those with .qweight)
        awq_bases = []
        for key in all_keys:
            if key.endswith('.qweight'):
                base = key.replace('.qweight', '')
                qzeros_key = base + '.qzeros'
                scales_key = base + '.scales'
                if qzeros_key in all_keys and scales_key in all_keys:
                    awq_bases.append(base)
        
        self.log(f"Found {len(awq_bases)} AWQ layers to dequantize")
        
        # Track which keys we've processed
        processed_keys = set()
        dequantized_count = 0
        
        with safe_open(model_path, framework="pt", device="cpu") as f:
            total_keys = len(all_keys)
            
            for i, key in enumerate(all_keys):
                if self._cancelled:
                    raise Exception("Cancelled")
                
                # Skip already processed AWQ components
                if key in processed_keys:
                    continue
                
                # Check if this is an AWQ layer
                is_awq_component = False
                awq_base = None
                
                for base in awq_bases:
                    if key.startswith(base + '.q') or key == base + '.scales':
                        is_awq_component = True
                        awq_base = base
                        break
                
                if is_awq_component and key.endswith('.qweight'):
                    # Dequantize this AWQ layer
                    qweight_key = awq_base + '.qweight'
                    qzeros_key = awq_base + '.qzeros'
                    scales_key = awq_base + '.scales'
                    
                    qweight = f.get_tensor(qweight_key).numpy()
                    qzeros = f.get_tensor(qzeros_key).numpy()
                    scales = f.get_tensor(scales_key).numpy()
                    
                    # Dequantize
                    weight_fp16 = self.dequantize_layer_numpy(qweight, qzeros, scales, group_size)
                    
                    # Output as .weight tensor
                    output_name = awq_base + '.weight'
                    output_callback(output_name, weight_fp16)
                    
                    # Mark all AWQ components as processed
                    processed_keys.add(qweight_key)
                    processed_keys.add(qzeros_key)
                    processed_keys.add(scales_key)
                    
                    dequantized_count += 1
                    
                    if dequantized_count % 10 == 0:
                        self.log(f"Dequantized {dequantized_count}/{len(awq_bases)} layers")
                    
                    # Clean up
                    del qweight, qzeros, scales, weight_fp16
                    gc.collect()
                    
                elif is_awq_component:
                    # Skip other AWQ components (qzeros, scales) - handled with qweight
                    continue
                else:
                    # Non-AWQ tensor - pass through
                    tensor = f.get_tensor(key)
                    if hasattr(tensor, 'numpy'):
                        arr = tensor.numpy()
                    else:
                        arr = np.array(tensor)
                    
                    # Convert to FP16 if needed
                    if arr.dtype in [np.float32, np.float64]:
                        arr = arr.astype(np.float16)
                    
                    output_callback(key, arr)
                    del tensor, arr
                
                # Progress update
                progress = int(100 * (i + 1) / total_keys)
                self.progress(progress, f"Processing {i+1}/{total_keys}")
                
                # Periodic GC
                if i % 50 == 0:
                    gc.collect()
        
        self.log(f"AWQ dequantization complete: {dequantized_count} layers")
        return dequantized_count
    
    def estimate_memory_usage(self, model_path: Path) -> Dict[str, float]:
        """
        Estimate memory usage for AWQ dequantization.
        
        Returns dict with:
        - awq_size_mb: Size of AWQ tensors
        - fp16_size_mb: Size after dequantization
        - peak_memory_mb: Estimated peak memory during processing
        """
        from safetensors import safe_open
        
        awq_size = 0
        fp16_size = 0
        
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                size = tensor.numel() * tensor.element_size()
                awq_size += size
                
                # Estimate FP16 size
                if key.endswith('.qweight'):
                    # qweight expands 8x (INT4 -> FP16)
                    fp16_size += tensor.numel() * 8 * 2  # 8x elements, 2 bytes each
                elif key.endswith('.qzeros') or key.endswith('.scales'):
                    # These are removed after dequantization
                    pass
                else:
                    # Non-AWQ tensors stay same size (or convert to FP16)
                    fp16_size += tensor.numel() * 2
        
        # Peak memory: need to hold qweight + qzeros + scales + output for one layer
        # Estimate based on largest layer (assume ~4096x4096 typical)
        peak_layer_size = 4096 * 4096 * 2 * 4  # 4 tensors worth
        
        return {
            'awq_size_mb': awq_size / (1024 * 1024),
            'fp16_size_mb': fp16_size / (1024 * 1024),
            'peak_memory_mb': peak_layer_size / (1024 * 1024) + 500  # +500MB overhead
        }


def is_awq_model(path: Path) -> bool:
    """Quick check if a file is an AWQ model."""
    from quantizer import detect_awq_model
    info = detect_awq_model(path)
    return info.is_awq
