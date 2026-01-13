"""
GGUF Converter - Quantization Module
Core quantization logic for converting AI models to GGUF format

Author: miha2017
"""

import struct
import json
import time
import gc
import traceback
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# LOGGING
# =============================================================================

# Get the module logger (will be configured by main module)
file_logger = logging.getLogger('gguf_converter')

def log_to_file(msg: str, level: str = "INFO"):
    """Log message to file with timestamp."""
    if level == "DEBUG":
        file_logger.debug(msg)
    elif level == "WARNING":
        file_logger.warning(msg)
    elif level == "ERROR":
        file_logger.error(msg)
    else:
        file_logger.info(msg)
    # Flush to ensure logs are written immediately
    for handler in file_logger.handlers:
        handler.flush()

def log_exception(msg: str):
    """Log exception with full traceback."""
    file_logger.error(f"{msg}\n{traceback.format_exc()}")


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class QuantType(Enum):
    """Quantization type enumeration with metadata."""
    Q3_K_S = ("Q3_K_S", "3-bit small", "~2.5 bits/weight")
    Q3_K_M = ("Q3_K_M", "3-bit medium", "~3 bits/weight")
    Q4_K_S = ("Q4_K_S", "4-bit small", "~4 bits/weight")
    Q4_K_M = ("Q4_K_M", "4-bit medium", "~4.5 bits/weight, recommended")
    Q5_K_S = ("Q5_K_S", "5-bit small", "~5 bits/weight")
    Q5_K_M = ("Q5_K_M", "5-bit medium", "~5.5 bits/weight")
    Q6_K = ("Q6_K", "6-bit", "~6 bits/weight")
    Q8_0 = ("Q8_0", "8-bit", "~8 bits/weight, near-lossless")
    
    @property
    def code(self): return self.value[0]
    @property
    def label(self): return self.value[1]
    @property
    def description(self): return self.value[2]


@dataclass
class ModelInfo:
    """Information about a model file."""
    path: Path
    model_type: str
    size_bytes: int
    num_files: int
    model_name: str = ""
    architecture: Optional[str] = None
    dtype: Optional[str] = None
    num_tensors: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ConversionResult:
    """Result of a model conversion operation."""
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    compression_ratio: float = 0.0


# =============================================================================
# MODEL ANALYZER
# =============================================================================

def extract_model_name(path: Path) -> str:
    """Extract model name from metadata or filename."""
    metadata_name = None
    
    if path.suffix.lower() == ".safetensors":
        try:
            with open(path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_size).decode('utf-8'))
                meta = header.get("__metadata__", {})
                for key in ["modelspec.title", "ss_output_name", "ss_sd_model_name", "model_name", "name", "title"]:
                    if key in meta and meta[key]:
                        value = str(meta[key]).strip()
                        if value and len(value) > 1:
                            metadata_name = value.replace(" ", "-").replace("/", "-").replace("\\", "-")
                            if metadata_name.lower().endswith(('.safetensors', '.ckpt', '.pt')):
                                metadata_name = Path(metadata_name).stem
                            break
        except: pass
    
    if metadata_name: return metadata_name
    
    config_name = _extract_name_from_config(path)
    if config_name: return config_name
    
    name = path.stem
    for suffix in ["-fp16", "-bf16", "-fp32", "-fp8", "_fp16", "_bf16", "_fp32", "_fp8", "-pruned", "_pruned", "-ema", "_ema"]:
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
    
    generic_names = ["diffusion_pytorch_model", "model", "pytorch_model", "unet", "vae", "text_encoder"]
    if name.lower() in [g.lower() for g in generic_names]:
        parent = path.parent.name
        if parent and parent.lower() != "downloads" and not parent.startswith("."):
            return parent
    
    return name if name else path.stem


def _extract_name_from_config(path: Path) -> Optional[str]:
    """Extract model name from config.json only for generic model filenames."""
    generic_names = ["diffusion_pytorch_model", "model", "pytorch_model", "unet", "vae", "text_encoder"]
    stem_lower = path.stem.lower()
    
    for suffix in ["-fp16", "-bf16", "-fp32", "-fp8", "_fp16", "_bf16", "_fp32", "_fp8", "-pruned", "_pruned", "-ema", "_ema"]:
        if stem_lower.endswith(suffix.lower()):
            stem_lower = stem_lower[:-len(suffix)]
    
    is_generic = any(stem_lower == g.lower() for g in generic_names)
    if not is_generic:
        return None
    
    config_files = ["config.json", "model_index.json"]
    name_keys = ["_name_or_path", "model_name", "name"]
    current = path.parent
    class_name_fallback = None
    
    for _ in range(4):
        for config_name in config_files:
            config_path = current / config_name
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for key in name_keys:
                        if key in data and data[key]:
                            value = str(data[key]).strip()
                            if "/" in value: value = value.split("/")[-1]
                            if value and len(value) > 1:
                                return value.replace(" ", "-").replace("/", "-").replace("\\", "-")
                    if "_class_name" in data and data["_class_name"] and not class_name_fallback:
                        class_name = str(data["_class_name"])
                        for suffix in ["Transformer3DModel", "Pipeline", "Model", "2DConditionModel", "3DModel", "Transformer", "UNet", "VAE", "Encoder", "Decoder"]:
                            if class_name.endswith(suffix) and len(class_name) > len(suffix):
                                class_name = class_name[:-len(suffix)]
                                break
                        if class_name and len(class_name) > 2:
                            class_name_fallback = class_name
                except: pass
        parent = current.parent
        if parent == current: break
        current = parent
    return class_name_fallback


class ModelAnalyzer:
    """Analyze model files to extract metadata."""
    
    @staticmethod
    def analyze(path: Path) -> ModelInfo:
        if not path.exists():
            return ModelInfo(path, "unknown", 0, 0, error="Path does not exist")
        if path.is_dir():
            return ModelAnalyzer._analyze_directory(path)
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            return ModelAnalyzer._analyze_safetensors(path)
        elif suffix in [".pt", ".pth", ".bin"]:
            return ModelAnalyzer._analyze_pytorch(path)
        elif suffix == ".gguf":
            return ModelInfo(path, "gguf", path.stat().st_size, 1, error="Already GGUF")
        return ModelInfo(path, "unknown", 0, 0, error=f"Unsupported: {suffix}")
    
    @staticmethod
    def _analyze_safetensors(path: Path) -> ModelInfo:
        try:
            size = path.stat().st_size
            with open(path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_size).decode('utf-8'))
            num_tensors = len([k for k in header.keys() if k != "__metadata__"])
            dtype = None
            for k, v in header.items():
                if k != "__metadata__" and isinstance(v, dict):
                    dtype = v.get("dtype")
                    break
            model_name = extract_model_name(path)
            return ModelInfo(path, "safetensors", size, 1, model_name=model_name, dtype=dtype, num_tensors=num_tensors)
        except Exception as e:
            return ModelInfo(path, "safetensors", path.stat().st_size, 1, model_name=path.stem, error=str(e))
    
    @staticmethod
    def _analyze_pytorch(path: Path) -> ModelInfo:
        try:
            import torch
            size = path.stat().st_size
            kw = {'map_location': 'cpu'}
            if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 13):
                kw['weights_only'] = False
            ckpt = torch.load(path, **kw)
            sd = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else {}
            dtype = None
            for v in sd.values():
                if hasattr(v, 'dtype'):
                    dtype = str(v.dtype).replace('torch.', '')
                    break
            model_name = extract_model_name(path)
            return ModelInfo(path, "pytorch", size, 1, model_name=model_name, dtype=dtype, num_tensors=len(sd))
        except Exception as e:
            return ModelInfo(path, "pytorch", path.stat().st_size, 1, model_name=path.stem, error=str(e))
    
    @staticmethod
    def _analyze_directory(path: Path) -> ModelInfo:
        files = list(path.rglob("*.safetensors")) + list(path.rglob("*.bin"))
        if not files:
            return ModelInfo(path, "unknown", 0, 0, error="No model files")
        size = sum(f.stat().st_size for f in files)
        return ModelInfo(path, "diffusers", size, len(files), model_name=path.name)


# =============================================================================
# QUANTIZATION CONFIGURATION
# =============================================================================

# Конфигурация типов квантизации: bits, max_val, block_bytes, group_size
QUANT_CONFIGS = {
    'Q3_K': {'bits': 3, 'max_val': 7, 'block_bytes': 100, 'group_size': 8},
    'Q4_K': {'bits': 4, 'max_val': 15, 'block_bytes': 132, 'group_size': 2},
    'Q5_K': {'bits': 5, 'max_val': 31, 'block_bytes': 164, 'group_size': 8},
    'Q6_K': {'bits': 6, 'max_val': 63, 'block_bytes': 196, 'group_size': 4},
}


# =============================================================================
# GGUF CONVERTER
# =============================================================================

class GGUFConverter:
    """Main converter class for quantizing models to GGUF format."""
    
    def __init__(self, progress_cb: Optional[Callable] = None, log_cb: Optional[Callable] = None, log_file: Optional[Path] = None):
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self._cancelled = False
        # File logging for crash debugging
        self.log_file = log_file or Path(__file__).parent / "converter.log"
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize log file with timestamp."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
        except Exception:
            pass
    
    def cancel(self):
        """Cancel the current conversion."""
        self._cancelled = True
    
    def log(self, msg: str):
        """Log message to UI and file."""
        if self.log_cb: 
            self.log_cb(msg)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass
    
    def progress(self, val: int, status: str):
        """Update progress bar."""
        if self.progress_cb: 
            self.progress_cb(val, status)
    
    def convert(self, info: ModelInfo, qtype: QuantType, out_dir: Path) -> ConversionResult:
        """Convert a model to GGUF format."""
        start = time.time()
        try:
            if info.error and "GGUF" in info.error:
                return ConversionResult(False, error_message=info.error)
            
            out_name = f"{info.model_name}-{qtype.code}.gguf"
            out_path = out_dir / out_name
            
            log_to_file(f"Input: {info.path.name}")
            log_to_file(f"Output: {out_name}")
            log_to_file(f"Quantization: {qtype.code}")
            
            self.log(f"Input: {info.path.name}")
            self.log(f"Output: {out_name}")
            self.log(f"Quantization: {qtype.code}")
            self.progress(0, "Preparing...")
            
            result = self._convert_python(info, qtype, out_path)
            if result.success:
                result.elapsed_time = time.time() - start
                result.input_size_mb = info.size_bytes / (1024*1024)
                result.output_size_mb = out_path.stat().st_size / (1024*1024)
                result.compression_ratio = result.input_size_mb / result.output_size_mb if result.output_size_mb > 0 else 0
                log_to_file(f"Success! Size: {result.input_size_mb:.1f}MB -> {result.output_size_mb:.1f}MB, Time: {result.elapsed_time:.1f}s")
            return result
        except Exception as e:
            log_exception(f"Convert error: {str(e)}")
            return ConversionResult(False, error_message=str(e))
    
    def _convert_python(self, info: ModelInfo, qtype: QuantType, out_path: Path) -> ConversionResult:
        """Потоковая конвертация - обрабатываем тензоры по одному для экономии памяти."""
        try:
            import torch
            import numpy as np
            
            self.progress(10, "Loading weights...")
            log_to_file(f"Loading model: {info.path.name}")
            self.log(f"Loading model: {info.path.name}")
            
            # Для safetensors используем потоковую обработку
            if info.model_type == "safetensors":
                return self._convert_safetensors_streaming(info, qtype, out_path)
            elif info.model_type == "pytorch":
                kw = {'map_location': 'cpu'}
                if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 13):
                    kw['weights_only'] = False
                ckpt = torch.load(info.path, **kw)
                tensors = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else {}
                log_to_file(f"Found {len(tensors)} tensors in pytorch model")
                self.log(f"Found {len(tensors)} tensors in pytorch model")
                del ckpt
                gc.collect()
            else:
                return ConversionResult(False, error_message="Unsupported type")
            
            # Для pytorch - старый метод
            self.progress(20, "Processing tensors...")
            log_to_file("Starting FP8 processing...")
            try:
                tensors = self._process_fp8_model(tensors)
            except Exception as e:
                error_msg = f"FP8 processing error: {type(e).__name__}: {str(e)}"
                log_exception(error_msg)
                self.log(f"ERROR: {error_msg}")
                return ConversionResult(False, error_message=error_msg)
            
            gc.collect()
            
            self.progress(30, "Quantizing...")
            log_to_file(f"Starting quantization with {qtype.code}, {len(tensors)} tensors...")
            self.log(f"Quantizing {len(tensors)} tensors with {qtype.code}...")
            
            try:
                self._write_gguf(tensors, out_path, qtype)
            except Exception as e:
                error_msg = f"GGUF write error: {type(e).__name__}: {str(e)}"
                log_exception(error_msg)
                self.log(f"ERROR: {error_msg}")
                return ConversionResult(False, error_message=error_msg)
            
            self.progress(100, "Done!")
            log_to_file("Conversion completed successfully!")
            self.log("Conversion completed successfully!")
            return ConversionResult(True, output_path=out_path)
        except MemoryError as e:
            error_msg = f"Out of memory: {str(e)}"
            log_exception(error_msg)
            self.log(f"ERROR: {error_msg}")
            return ConversionResult(False, error_message=error_msg)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            log_exception(f"Conversion error: {error_msg}")
            self.log(f"ERROR: {error_msg}")
            return ConversionResult(False, error_message=error_msg)

    
    def _convert_safetensors_streaming(self, info: ModelInfo, qtype: QuantType, out_path: Path) -> ConversionResult:
        """Потоковая конвертация safetensors - с многопоточной квантизацией."""
        import numpy as np
        import torch
        from safetensors import safe_open
        from concurrent.futures import ThreadPoolExecutor
        import os
        
        # Используем половину ядер для квантизации (остальные для I/O и системы)
        num_workers = max(1, os.cpu_count() // 2)
        log_to_file(f"Using {num_workers} workers for parallel quantization")
        
        try:
            log_to_file("Running in CPU mode (optimized for quantization)")
            
            # Сначала собираем информацию о тензорах
            with safe_open(info.path, framework="pt", device="cpu") as f:
                all_keys = list(f.keys())
            
            total = len(all_keys)
            log_to_file(f"Found {total} tensors in safetensors (streaming mode)")
            self.log(f"Found {total} tensors (streaming mode)")
            
            # Паттерны служебных тензоров FP8
            skip_patterns = ['comfy_quant', 'weight_scale', 'input_scale', 'output_scale', 
                             '_scale', '_zero_point', 'quant_state']
            
            # Собираем scale тензоры для FP8 деквантизации
            scale_tensors = {}
            with safe_open(info.path, framework="pt", device="cpu") as f:
                for name in all_keys:
                    if 'weight_scale' in name:
                        base_name = name.replace('.weight_scale', '.weight')
                        scale_tensors[base_name] = f.get_tensor(name)
            
            log_to_file(f"Found {len(scale_tensors)} scale tensors for FP8")
            
            # Фильтруем ключи - убираем служебные
            tensor_keys = [k for k in all_keys if not any(p in k for p in skip_patterns)]
            num_tensors = len(tensor_keys)
            log_to_file(f"Tensors to process: {num_tensors}")
            self.log(f"Tensors to process: {num_tensors}")
            
            # Функция квантизации для параллельного выполнения
            def quantize_tensor(args):
                name, arr, qtype_code = args
                if self._should_quantize(name, arr.shape):
                    layer_qtype = self._get_quant_type_for_layer(name, qtype_code, arr.shape)
                    try:
                        qdata, dtype_code = self._quantize(arr, layer_qtype)
                        return (name, qdata, arr.shape, dtype_code, True)
                    except Exception as e:
                        return (name, arr.astype(np.float32).tobytes(), arr.shape, 0, False)
                else:
                    return (name, arr.astype(np.float32).tobytes(), arr.shape, 0, False)
            
            # Открываем выходной файл
            with open(out_path, 'wb') as out_f:
                # GGUF Header
                out_f.write(struct.pack('<I', 0x46554747))  # Magic: "GGUF"
                out_f.write(struct.pack('<I', 3))           # Version
                out_f.write(struct.pack('<Q', num_tensors)) # Num tensors
                out_f.write(struct.pack('<Q', 0))           # Num KV pairs
                
                # Обрабатываем батчами для параллельной квантизации
                tensor_data = []
                quantized_count = 0
                f32_count = 0
                batch_size = max(8, num_workers * 2)
                
                # Создаём executor один раз для всей конвертации
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    with safe_open(info.path, framework="pt", device="cpu") as f:
                        i = 0
                        while i < num_tensors:
                            if self._cancelled:
                                raise Exception("Cancelled")
                            
                            # Загружаем батч тензоров
                            batch_end = min(i + batch_size, num_tensors)
                            batch_tasks = []
                            
                            for j in range(i, batch_end):
                                name = tensor_keys[j]
                                tensor = f.get_tensor(name)
                                
                                # FP8 деквантизация
                                if name in scale_tensors:
                                    try:
                                        scale = scale_tensors[name]
                                        tensor = tensor.float() * scale.float()
                                    except Exception as e:
                                        log_to_file(f"Warning: FP8 dequant failed for {name}: {e}")
                                
                                # Конвертируем в numpy
                                try:
                                    if tensor.dtype == torch.float32:
                                        arr = tensor.numpy().copy()
                                    elif tensor.dtype in (torch.float16, torch.bfloat16):
                                        arr = tensor.float().numpy()
                                    else:
                                        arr = tensor.float().numpy()
                                except:
                                    arr = np.array(tensor.detach().cpu().float(), dtype=np.float32)
                                
                                del tensor
                                batch_tasks.append((name, arr, qtype.code))
                            
                            # Параллельная квантизация батча
                            results = list(executor.map(quantize_tensor, batch_tasks))
                            
                            # Собираем результаты
                            for name, qdata, shape, dtype_code, was_quantized in results:
                                tensor_data.append((name, qdata, shape, dtype_code))
                                if was_quantized:
                                    quantized_count += 1
                                else:
                                    f32_count += 1
                            
                            # Обновляем прогресс
                            progress = 10 + int(80 * batch_end / num_tensors)
                            self.progress(progress, f"Tensor {batch_end}/{num_tensors}")
                            
                            # Логируем каждые 100 тензоров
                            if batch_end % 100 < batch_size:
                                log_to_file(f"Processing tensor {batch_end}/{num_tensors}")
                                self.log(f"Processing: {batch_end}/{num_tensors}")
                            
                            i = batch_end
                            
                            # Периодическая очистка памяти
                            if batch_end % 500 < batch_size:
                                gc.collect()
                
                log_to_file(f"Quantized: {quantized_count}, F32: {f32_count}")
                self.log(f"Quantized: {quantized_count}, F32: {f32_count}")
                
                # Записываем заголовки тензоров
                self.log("Writing tensor headers...")
                offset = 0
                for name, data, shape, dtype_code in tensor_data:
                    nb = name.encode('utf-8')
                    out_f.write(struct.pack('<Q', len(nb)))
                    out_f.write(nb)
                    out_f.write(struct.pack('<I', len(shape)))
                    for d in shape:
                        out_f.write(struct.pack('<Q', d))
                    out_f.write(struct.pack('<I', dtype_code))
                    out_f.write(struct.pack('<Q', offset))
                    offset += len(data)
                
                # Записываем данные
                self.log("Writing tensor data...")
                for name, data, shape, dtype_code in tensor_data:
                    out_f.write(data)
                
                # Освобождаем память
                del tensor_data
                del scale_tensors
                gc.collect()
            
            self.progress(100, "Done!")
            log_to_file("Streaming conversion completed successfully!")
            self.log("Conversion completed successfully!")
            return ConversionResult(True, output_path=out_path)
            
        except MemoryError as e:
            error_msg = f"Out of memory: {str(e)}"
            log_exception(error_msg)
            self.log(f"ERROR: {error_msg}")
            return ConversionResult(False, error_message=error_msg)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            log_exception(f"Streaming conversion error: {error_msg}")
            self.log(f"ERROR: {error_msg}")
            return ConversionResult(False, error_message=error_msg)

    
    def _process_fp8_model(self, tensors):
        """Обработка FP8 моделей: деквантизация весов и фильтрация служебных тензоров."""
        import numpy as np
        
        total = len(tensors)
        log_to_file(f"Processing {total} tensors for FP8...")
        self.log(f"Processing {total} tensors...")
        
        # Паттерны служебных тензоров FP8/ComfyUI, которые нужно пропустить
        skip_patterns = ['comfy_quant', 'weight_scale', 'input_scale', 'output_scale', 
                         '_scale', '_zero_point', 'quant_state']
        
        # Собираем scale тензоры для деквантизации
        log_to_file("Collecting scale tensors...")
        scale_tensors = {}
        try:
            for name, tensor in tensors.items():
                if 'weight_scale' in name:
                    base_name = name.replace('.weight_scale', '.weight')
                    scale_tensors[base_name] = tensor
        except Exception as e:
            log_to_file(f"Error collecting scale tensors: {e}")
            raise
        
        log_to_file(f"Found {len(scale_tensors)} scale tensors for FP8 dequantization")
        
        processed = {}
        skipped_count = 0
        dequant_count = 0
        
        log_to_file("Starting tensor processing loop...")
        
        # Конвертируем items в список чтобы избежать проблем с итерацией
        tensor_items = list(tensors.items())
        log_to_file(f"Converted to list: {len(tensor_items)} items")
        
        for i, (name, tensor) in enumerate(tensor_items):
            try:
                # Пропускаем служебные тензоры
                if any(pattern in name for pattern in skip_patterns):
                    skipped_count += 1
                    continue
                
                # Проверяем, нужна ли деквантизация FP8
                if name in scale_tensors:
                    scale = scale_tensors[name]
                    try:
                        if hasattr(tensor, 'float'):
                            tensor = tensor.float() * scale.float()
                            dequant_count += 1
                    except Exception as e:
                        log_to_file(f"Warning: Failed to dequantize {name}: {e}")
                        self.log(f"Warning: Failed to dequantize {name}: {e}")
                
                processed[name] = tensor
                
                # Логируем прогресс каждые 500 тензоров
                if (i + 1) % 500 == 0:
                    log_to_file(f"FP8 processing: {i+1}/{total} tensors...")
                    self.log(f"FP8 processing: {i+1}/{total}")
                    gc.collect()
                    
            except Exception as e:
                log_to_file(f"Error processing tensor {i} ({name}): {type(e).__name__}: {e}")
                self.log(f"Error processing tensor {name}: {e}")
                continue
        
        log_to_file(f"FP8 processing complete: {len(processed)} tensors to quantize, {skipped_count} skipped, {dequant_count} dequantized")
        self.log(f"Tensors to quantize: {len(processed)}")
        if skipped_count > 0:
            self.log(f"Skipped {skipped_count} service tensors")
        if dequant_count > 0:
            self.log(f"Dequantized {dequant_count} FP8 weights")
        
        del tensor_items
        gc.collect()
        
        return processed
    
    def _should_quantize(self, name: str, shape: tuple) -> bool:
        """Определяет, нужно ли квантизировать тензор или оставить в F32."""
        numel = 1
        for d in shape:
            numel *= d
        
        # Тензоры меньше 256 элементов - F32
        if numel < 256:
            return False
        
        # Bias тензоры - F32
        if '.bias' in name:
            return False
        
        # Нормализация - F32
        norm_patterns = ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']
        if any(p in name.lower() for p in norm_patterns):
            if numel < 10000:
                return False
        
        # Embedding тензоры небольшого размера - F32
        if 'embed' in name.lower() and numel < 100000:
            return False
        
        return True
    
    def _get_quant_type_for_layer(self, name: str, qtype_code: str, shape: tuple) -> str:
        """Определяет тип квантизации для конкретного слоя (Q4_K_M использует смешанную квантизацию)."""
        if qtype_code != "Q4_K_M":
            return qtype_code
        
        # Q4_K_M: важные слои в Q6_K, остальные в Q4_K
        important_patterns = [
            'to_out', 'to_add_out', 'proj_out', 'output',
            'attn.to_q', 'attn.to_k', 'attn.to_v',
            'blocks.0.', 'blocks.1.',
        ]
        
        numel = 1
        for d in shape:
            numel *= d
        
        # Очень большие тензоры (>100M параметров) - Q4_K для экономии памяти
        if numel > 100_000_000:
            return "Q4_K"
        
        # Важные слои - Q6_K для лучшего качества
        if any(p in name for p in important_patterns):
            return "Q6_K"
        
        return "Q4_K"

    
    def _write_gguf(self, tensors, out_path: Path, qtype: QuantType):
        """Write tensors to GGUF file."""
        import numpy as np
        
        filtered_tensors = [(name, tensor) for name, tensor in tensors.items()]
        total = len(filtered_tensors)
        
        log_to_file(f"Writing GGUF with {total} tensors to {out_path}")
        self.log(f"Writing GGUF with {total} tensors...")
        
        with open(out_path, 'wb') as f:
            # GGUF Header
            f.write(struct.pack('<I', 0x46554747))  # Magic: "GGUF"
            f.write(struct.pack('<I', 3))           # Version
            f.write(struct.pack('<Q', total))       # Num tensors
            f.write(struct.pack('<Q', 0))           # Num KV pairs (metadata)
            
            data_list = []
            quantized_count = 0
            f32_count = 0
            
            for i, (name, tensor) in enumerate(filtered_tensors):
                if self._cancelled: 
                    raise Exception("Cancelled")
                
                log_to_file(f"[{i+1}/{total}] {name} shape={getattr(tensor, 'shape', 'unknown')}")
                
                if i % 10 == 0 or i == total - 1:
                    self.log(f"Tensor {i+1}/{total}")
                
                self.progress(30 + int(60 * i / total), f"Tensor {i+1}/{total}")
                
                import numpy as np
                try:
                    arr = tensor.float().numpy() if hasattr(tensor, 'numpy') else np.array(tensor, dtype=np.float32)
                except Exception as e:
                    log_to_file(f"Warning: Failed to convert {name}: {e}", "WARNING")
                    self.log(f"Warning: Failed to convert {name}: {e}")
                    arr = np.array(tensor.detach().cpu().float()) if hasattr(tensor, 'detach') else np.array(tensor, dtype=np.float32)
                
                if self._should_quantize(name, arr.shape):
                    layer_qtype = self._get_quant_type_for_layer(name, qtype.code, arr.shape)
                    try:
                        qdata, dtype_code = self._quantize(arr, layer_qtype)
                        quantized_count += 1
                    except MemoryError:
                        log_exception(f"Out of memory while quantizing {name} (shape={arr.shape})")
                        raise RuntimeError(f"Out of memory while quantizing {name}")
                    except Exception as e:
                        log_exception(f"Quantization error for tensor {name} (shape={arr.shape}, qtype={layer_qtype})")
                        raise RuntimeError(f"Failed to quantize {name}: {e}")
                else:
                    qdata, dtype_code = arr.astype(np.float32).tobytes(), 0
                    f32_count += 1
                
                data_list.append((name, qdata, arr.shape, dtype_code))
            
            log_to_file(f"Quantized: {quantized_count}, F32: {f32_count}")
            self.log(f"Quantized: {quantized_count}, F32: {f32_count}")
            self.log("Writing tensor headers...")
            
            # Записываем заголовки тензоров
            offset = 0
            for name, data, shape, dtype_code in data_list:
                nb = name.encode('utf-8')
                f.write(struct.pack('<Q', len(nb)))
                f.write(nb)
                f.write(struct.pack('<I', len(shape)))
                for d in shape: 
                    f.write(struct.pack('<Q', d))
                f.write(struct.pack('<I', dtype_code))
                f.write(struct.pack('<Q', offset))
                offset += len(data)
            
            self.log("Writing tensor data...")
            
            # Записываем данные тензоров
            for _, data, _, _ in data_list:
                f.write(data)
    
    def _quantize(self, arr, qtype: str):
        """Квантизация тензора в указанный формат (CPU-only для максимальной скорости)."""
        import numpy as np
        
        try:
            if qtype == "Q8_0":
                # Q8_0: 8-bit квантизация с одним scale на тензор
                scale = max(np.max(np.abs(arr)), 1e-10) / 127.0
                q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
                return struct.pack('<f', scale) + q.tobytes(), 8
            
            elif qtype in ["Q4_K_S", "Q4_K", "Q4_K_M"]:
                return self._quantize_q4_k(arr), 12
            
            elif qtype in ["Q6_K"]:
                return self._quantize_q6_k(arr), 14
            
            elif qtype in ["Q5_K_S", "Q5_K", "Q5_K_M"]:
                return self._quantize_q5_k(arr), 13
            
            elif qtype in ["Q3_K_S", "Q3_K", "Q3_K_M"]:
                return self._quantize_q3_k(arr), 11
            
            # По умолчанию - F16
            return arr.astype(np.float16).tobytes(), 1
        except Exception as e:
            log_exception(f"Quantization error ({qtype}, shape={arr.shape})")
            self.log(f"Quantization error ({qtype}): {e}, falling back to F32")
            return arr.astype(np.float32).tobytes(), 0

    
    # =========================================================================
    # BIT PACKING FUNCTIONS
    # =========================================================================
    
    def _pack_3bit(self, q, num_blocks):
        """Упаковка 8 значений по 3 бита в 3 байта."""
        import numpy as np
        q_reshaped = q.reshape(num_blocks, -1, 8)
        b0 = (q_reshaped[:, :, 0] & 0x07) | ((q_reshaped[:, :, 1] & 0x07) << 3) | ((q_reshaped[:, :, 2] & 0x03) << 6)
        b1 = ((q_reshaped[:, :, 2] >> 2) & 0x01) | ((q_reshaped[:, :, 3] & 0x07) << 1) | ((q_reshaped[:, :, 4] & 0x07) << 4) | ((q_reshaped[:, :, 5] & 0x01) << 7)
        b2 = ((q_reshaped[:, :, 5] >> 1) & 0x03) | ((q_reshaped[:, :, 6] & 0x07) << 2) | ((q_reshaped[:, :, 7] & 0x07) << 5)
        return np.stack([b0, b1, b2], axis=2).astype(np.uint8).reshape(num_blocks, -1)
    
    def _pack_4bit(self, q, num_blocks):
        """Упаковка 2 значений по 4 бита в 1 байт."""
        import numpy as np
        return (q[:, ::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    
    def _pack_5bit(self, q, num_blocks):
        """Упаковка 8 значений по 5 бит в 5 байт."""
        import numpy as np
        q_reshaped = q.reshape(num_blocks, -1, 8)
        b0 = (q_reshaped[:, :, 0] & 0x1F) | ((q_reshaped[:, :, 1] & 0x07) << 5)
        b1 = ((q_reshaped[:, :, 1] >> 3) & 0x03) | ((q_reshaped[:, :, 2] & 0x1F) << 2) | ((q_reshaped[:, :, 3] & 0x01) << 7)
        b2 = ((q_reshaped[:, :, 3] >> 1) & 0x0F) | ((q_reshaped[:, :, 4] & 0x0F) << 4)
        b3 = ((q_reshaped[:, :, 4] >> 4) & 0x01) | ((q_reshaped[:, :, 5] & 0x1F) << 1) | ((q_reshaped[:, :, 6] & 0x03) << 6)
        b4 = ((q_reshaped[:, :, 6] >> 2) & 0x07) | ((q_reshaped[:, :, 7] & 0x1F) << 3)
        return np.stack([b0, b1, b2, b3, b4], axis=2).astype(np.uint8).reshape(num_blocks, -1)
    
    def _pack_6bit(self, q, num_blocks):
        """Упаковка 4 значений по 6 бит в 3 байта."""
        import numpy as np
        q_reshaped = q.reshape(num_blocks, -1, 4)
        b0 = (q_reshaped[:, :, 0] & 0x3F) | ((q_reshaped[:, :, 1] & 0x03) << 6)
        b1 = ((q_reshaped[:, :, 1] >> 2) & 0x0F) | ((q_reshaped[:, :, 2] & 0x0F) << 4)
        b2 = ((q_reshaped[:, :, 2] >> 4) & 0x03) | ((q_reshaped[:, :, 3] & 0x3F) << 2)
        return np.stack([b0, b1, b2], axis=2).astype(np.uint8).reshape(num_blocks, -1)
    
    # =========================================================================
    # UNIFIED BLOCK QUANTIZATION
    # =========================================================================
    
    def _quantize_block(self, arr, qtype: str):
        """Универсальная блочная квантизация для Q3_K, Q4_K, Q5_K, Q6_K."""
        import numpy as np
        
        config = QUANT_CONFIGS[qtype]
        max_val = config['max_val']
        block_bytes = config['block_bytes']
        block_size = 256
        
        try:
            flat = arr.flatten()
            if flat.dtype != np.float32:
                flat = flat.astype(np.float32)
            
            # Паддинг до кратного block_size
            pad_size = (block_size - len(flat) % block_size) % block_size
            if pad_size > 0:
                flat = np.concatenate([flat, np.zeros(pad_size, dtype=np.float32)])
            
            num_blocks = len(flat) // block_size
            blocks = flat.reshape(num_blocks, block_size)
            
            # Вычисляем min/max/scale для всех блоков
            block_mins = blocks.min(axis=1)
            block_maxs = blocks.max(axis=1)
            block_ranges = block_maxs - block_mins
            scales = np.where(block_ranges > 1e-10, block_ranges / float(max_val), 1e-10)
            
            # Квантизуем все блоки
            normalized = (blocks - block_mins[:, None]) / scales[:, None]
            q = np.clip(np.round(normalized), 0, max_val).astype(np.uint8)
            
            # Упаковка битов в зависимости от типа
            if qtype == 'Q3_K':
                packed = self._pack_3bit(q, num_blocks)
            elif qtype == 'Q4_K':
                packed = self._pack_4bit(q, num_blocks)
            elif qtype == 'Q5_K':
                packed = self._pack_5bit(q, num_blocks)
            else:  # Q6_K
                packed = self._pack_6bit(q, num_blocks)
            
            # Собираем результат: scale(2) + min(2) + packed
            packed_size = block_bytes - 4
            result = bytearray(num_blocks * block_bytes)
            scales_f16 = scales.astype(np.float16)
            mins_f16 = block_mins.astype(np.float16)
            
            for i in range(num_blocks):
                offset = i * block_bytes
                result[offset:offset+2] = scales_f16[i].tobytes()
                result[offset+2:offset+4] = mins_f16[i].tobytes()
                result[offset+4:offset+block_bytes] = packed[i].tobytes()
            
            return bytes(result)
        except Exception as e:
            log_exception(f"{qtype} quantization error: shape={arr.shape}")
            raise
    
    # =========================================================================
    # QUANTIZATION WRAPPERS (for backward compatibility)
    # =========================================================================
    
    def _quantize_q3_k(self, arr):
        """Q3_K квантизация: 3-bit с блочным scale."""
        return self._quantize_block(arr, 'Q3_K')
    
    def _quantize_q4_k(self, arr):
        """Q4_K квантизация: 4-bit с блочным scale."""
        return self._quantize_block(arr, 'Q4_K')
    
    def _quantize_q5_k(self, arr):
        """Q5_K квантизация: 5-bit с блочным scale."""
        return self._quantize_block(arr, 'Q5_K')
    
    def _quantize_q6_k(self, arr):
        """Q6_K квантизация: 6-bit с блочным scale."""
        return self._quantize_block(arr, 'Q6_K')
