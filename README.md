# GGUF Converter

<p align="center">
  <img src="images/logoLLM.png" alt="GGUF Converter Logo" width="160">
</p>

<p align="center">
  <img src="images/screenshot.png" alt="GGUF Converter Interface" width="600">
</p>

Universal utility for converting AI models to GGUF format with quantization.

[🇷🇺 Русская версия](README_RU.md)

## Features

- 🔄 Convert models from `.safetensors`, `.pt`, `.pth`, `.bin` formats
- 📊 8 quantization levels (Q3_K_S to Q8_0)
- 📁 Automatic Downloads folder scanning
- 🔍 Smart model name extraction from metadata
- 📂 **Custom output folder** (persisted between sessions)
- 📦 **Batch conversion** of multiple files
- 🔎 **GGUF file inspection** (view tensors, export to CSV)
- 🌍 Multilingual: English, Русский, 中文
- ⚡ FP8 model processing (automatic dequantization)
- 🎯 Mixed quantization Q4_K_M (important layers in Q6_K)

## Installation

### Requirements

- Python 3.8+
- Windows 10/11

### Dependencies

```bash
pip install numpy pillow safetensors pywin32
```

Optional for PyTorch models:
```bash
pip install torch
```

## Usage

### Launch

```bash
python gguf_converter.py
```

Or use `run_converter.bat` for quick launch.

### Conversion Process

1. Select a model from the list (Downloads folder is scanned automatically)
2. Or click "Browse" to select a file manually
3. Choose output folder (defaults to Downloads)
4. Select quantization level
5. Click "Convert"
6. The converted file will appear in the selected folder

### Batch Conversion

1. Enable "Batch mode" checkbox
2. Select multiple files in the dialog
3. Click "Convert"
4. All files will be processed sequentially
5. A summary will appear at the end

### GGUF File Inspection

1. Click "Inspect GGUF" button
2. Select a GGUF file
3. View tensor information:
   - Name, shape, data type, size
   - Total tensor count
   - GGUF format version
4. Export to CSV or copy to clipboard

## Quantization Levels

| Type | Description | Size |
|------|-------------|------|
| Q3_K_S | 3-bit small | ~2.5 bits/weight |
| Q3_K_M | 3-bit medium | ~3 bits/weight |
| Q4_K_S | 4-bit small | ~4 bits/weight |
| Q4_K_M | 4-bit medium | ~4.5 bits/weight ⭐ |
| Q5_K_S | 5-bit small | ~5 bits/weight |
| Q5_K_M | 5-bit medium | ~5.5 bits/weight |
| Q6_K | 6-bit | ~6 bits/weight |
| Q8_0 | 8-bit | ~8 bits/weight |

⭐ Q4_K_M — recommended balance of quality and size (uses mixed quantization: important layers in Q6_K)

## Project Structure

```
gguf-converter-v1/
├── gguf_converter.py    # Main script
├── quantizer.py         # Quantization logic
├── ui_widgets.py        # UI components
├── translations.py      # Translations (RU/EN/ZH)
├── settings.json        # User settings
├── run_converter.bat    # Windows launcher
├── images/
│   ├── logoLLM.png      # Logo
│   ├── logoLLM.ico      # Window icon
│   └── nayan.gif        # Nyan Cat animation
├── music/               # Music folder
│   └── *.mp3            # MP3 files for background
├── tools/               # Analysis utilities
│   ├── analyze_source_model.py    # Safetensors analysis
│   ├── check_quantization.py      # Quantization check
│   ├── compare_gguf_models.py     # GGUF comparison
│   ├── compare_tensors_detailed.py # Detailed comparison
│   ├── compare_two_models.py      # Two model comparison
│   └── test_quantization_types.py # Quantization tests
└── README.md            # Documentation
```

## Music

Place MP3 or WAV files in the `music/` folder for background music.
The 🔊 button toggles sound on/off. Music plays only during conversion.

## Features

### Double Launch Protection
The utility uses Windows Mutex to prevent running multiple instances.

### Smart Model Name Detection
Output filename is extracted in priority order:
1. **Safetensors metadata** (highest priority):
   - `modelspec.title` — standard specification
   - `ss_output_name` — Kohya trainer
   - `ss_sd_model_name` — Kohya SD model
   - `model_name`, `name`, `title` — common keys
2. **config.json** — for generic files (model.safetensors)
3. **Cleaned filename** — suffixes like `-fp16`, `_pruned` are removed
4. **Folder name** — only if file has generic name

### FP8 Model Processing
- Automatic FP8 weight detection
- FP8 → FP32 dequantization before quantization
- Service tensor filtering (scale, zero_point)

### Mixed Quantization (Q4_K_M)
Q4_K_M uses intelligent quantization:
- Important layers (attention, first blocks) → Q6_K
- Other layers → Q4_K
- Bias and normalization → F32

### Dark Theme
The interface uses a dark color scheme, comfortable for the eyes.
Dark window title bar is supported on Windows 11.

## Changelog

### v1.8
- ⚡ Progress bar animation optimization (3x faster)
- 🔧 Pre-created canvas elements instead of delete/create cycle
- 🎵 Fixed music playback - now random track after completion
- 🐛 Fixed percentage cutoff on the right
- 🎬 Independent animation thread - smooth animation regardless of CPU load

### v1.7
- ⚡ Multi-threaded quantization (uses all CPU cores)
- ⚡ Vectorized quantization functions (5-12x speedup)
- 🔧 Async UI updates via queue
- 🐛 Removed torch.cuda.empty_cache() calls (were slowing down)

### v1.2
- ✨ Output folder selection (saved in settings)
- ✨ Batch conversion of multiple files
- ✨ GGUF file inspection (view tensors, export CSV)
- 🐛 Fixed crash on Q3 quantization
- 🐛 Fixed bit packing in Q5_K and Q6_K
- 📝 Added tensor count to log output

### v1.1
- ✨ Multilingual support (RU/EN/ZH)
- ✨ FP8 model processing
- ✨ Mixed quantization Q4_K_M
- 🎨 Improved interface

### v1.0
- 🎉 Initial release

## License

MIT License

## Author

miha2017

---

## 💰 Donate

If you find this project useful, you can support development:

**Cryptocurrency:**

<p align="center">
  <img src="images/QR-tron-donate.png" alt="USDT TRC20 QR Code" width="150">
</p>

**USDT (TRC20):** `TFZoJGcYd8z2QPokiZSBcZnrkTevEnxpyR`

---

## ⚡ Contact:

https://t.me/mnbabeshko

*GGUF Converter v1.8*
