#!/usr/bin/env python3
"""
GGUF Converter - Universal Model Quantization Tool
Standalone utility for converting AI models to GGUF format

Author: miha2017
Version: 1.8
License: MIT
"""

import os
import sys
import time
import json
import struct
import random
import threading
import math
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from queue import Queue, Empty

# =============================================================================
# HIDE CONSOLE WINDOW (Windows)
# =============================================================================

def hide_console():
    """Hide the console window on Windows."""
    try:
        import ctypes
        # Get console window handle
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            # SW_HIDE = 0, SW_MINIMIZE = 6
            user32.ShowWindow(hwnd, 0)  # Hide console
    except Exception:
        pass

# Hide console immediately on import
hide_console()

# =============================================================================
# GPU DETECTION (for info only, not used for quantization)
# =============================================================================

GPU_AVAILABLE = False
GPU_NAME = "CPU"

def init_gpu():
    """Detect GPU for info display (not used for quantization - CPU is faster)."""
    global GPU_AVAILABLE, GPU_NAME
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_NAME = torch.cuda.get_device_name(0)
            return True
    except Exception:
        pass
    return False

# Detect GPU at module load (for info only)
init_gpu()

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DOWNLOADS_FOLDER = Path.home() / "Downloads"
MUSIC_FOLDER = SCRIPT_DIR / "music"
IMAGES_FOLDER = SCRIPT_DIR / "images"
LOGO_PATH = IMAGES_FOLDER / "logoLLM.png"
ICON_PATH = IMAGES_FOLDER / "logoLLM.ico"
NYAN_PATH = IMAGES_FOLDER / "nayan.gif"
SETTINGS_PATH = SCRIPT_DIR / "settings.json"
LOG_PATH = SCRIPT_DIR / "converter.log"

# Setup file logging
file_logger = logging.getLogger('gguf_converter')
file_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
file_logger.addHandler(file_handler)

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

# Log session start
log_to_file(f"\n{'='*60}\nSession started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas

# Import translations
from translations import TRANSLATIONS, get_text

# Import UI widgets
from ui_widgets import COLORS, BUTTON_RADIUS, BUTTON_PADDING_X, BUTTON_HEIGHT
from ui_widgets import make_round_button, make_icon_button, show_dark_dialog

# Import quantization module
from quantizer import (
    QuantType, ModelInfo, ConversionResult,
    ModelAnalyzer, GGUFConverter, QUANT_CONFIGS,
    extract_model_name, _extract_name_from_config
)

# Check for single instance (Windows only)
try:
    import win32event
    import win32api
    import winerror
    
    mutex = win32event.CreateMutex(None, False, 'GGUFConverterMutex')
    last_error = win32api.GetLastError()
    
    if last_error == winerror.ERROR_ALREADY_EXISTS:
        warn_root = tk.Tk()
        warn_root.withdraw()
        dialog = tk.Toplevel(warn_root)
        dialog.title("GGUF Converter")
        dialog.configure(bg='#3a3a3a')
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 150) // 2
        dialog.geometry(f"400x150+{x}+{y}")
        tk.Label(dialog, text="⚠", font=("Segoe UI", 32), fg='#f0932b', bg='#3a3a3a').pack(pady=(15, 5))
        tk.Label(dialog, text="GGUF Converter уже запущен!", font=("Segoe UI", 11, "bold"), fg='#e0e0e0', bg='#3a3a3a').pack()
        tk.Label(dialog, text="Закройте предыдущий экземпляр программы.", font=("Segoe UI", 9), fg='#959595', bg='#3a3a3a').pack(pady=(5, 10))
        btn = tk.Button(dialog, text="OK", command=lambda: sys.exit(0), font=("Segoe UI", 10), fg='#e0e0e0', bg='#4a4a4a',
                       activebackground='#5a5a5a', activeforeground='#e0e0e0', relief='flat', bd=0, padx=30, pady=5, cursor='hand2')
        btn.pack(pady=(0, 15))
        dialog.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
        dialog.mainloop()
        sys.exit(0)
except ImportError:
    pass

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# =============================================================================
# SETTINGS
# =============================================================================

def load_settings() -> dict:
    """Load settings from JSON file."""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"language": "ru", "output_folder": str(DOWNLOADS_FOLDER)}

def save_settings(settings: dict):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except:
        pass


# =============================================================================
# GGUF INSPECTOR
# =============================================================================

class GGUFInspector:
    """Inspect GGUF file contents."""
    
    GGUF_MAGIC = 0x46554747  # "GGUF"
    
    # GGUF data types
    DTYPE_NAMES = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "I8", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "I16", 17: "I32",
        18: "I64", 19: "F64", 20: "BF16"
    }
    
    DTYPE_SIZES = {
        0: 4, 1: 2, 2: 0.5, 3: 0.5625, 4: 0.5, 5: 0.5625,
        6: 0.5625, 7: 0.625, 8: 1, 9: 1.0625, 10: 1, 11: 0.4375,
        12: 0.5625, 13: 0.6875, 14: 0.8125, 15: 1.0625, 16: 2, 17: 4,
        18: 8, 19: 8, 20: 2
    }
    
    @staticmethod
    def inspect(path: Path) -> dict:
        """Inspect GGUF file and return info."""
        result = {
            "valid": False,
            "version": 0,
            "num_tensors": 0,
            "num_kv": 0,
            "tensors": [],
            "metadata": {},
            "total_size": 0,
            "error": None
        }
        
        try:
            with open(path, 'rb') as f:
                # Read magic
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != GGUFInspector.GGUF_MAGIC:
                    result["error"] = "Not a valid GGUF file"
                    return result
                
                # Read version
                result["version"] = struct.unpack('<I', f.read(4))[0]
                
                # Read tensor count and KV count
                result["num_tensors"] = struct.unpack('<Q', f.read(8))[0]
                result["num_kv"] = struct.unpack('<Q', f.read(8))[0]
                
                # Skip KV pairs (simplified - just read tensor info)
                # For full KV parsing we'd need more complex logic
                
                # Read tensor headers
                tensors = []
                for _ in range(result["num_tensors"]):
                    try:
                        # Read name
                        name_len = struct.unpack('<Q', f.read(8))[0]
                        name = f.read(name_len).decode('utf-8')
                        
                        # Read dimensions
                        n_dims = struct.unpack('<I', f.read(4))[0]
                        dims = []
                        for _ in range(n_dims):
                            dims.append(struct.unpack('<Q', f.read(8))[0])
                        
                        # Read dtype and offset
                        dtype = struct.unpack('<I', f.read(4))[0]
                        offset = struct.unpack('<Q', f.read(8))[0]
                        
                        # Calculate size
                        numel = 1
                        for d in dims:
                            numel *= d
                        dtype_size = GGUFInspector.DTYPE_SIZES.get(dtype, 4)
                        size_bytes = int(numel * dtype_size)
                        
                        tensors.append({
                            "name": name,
                            "shape": tuple(dims),
                            "dtype": GGUFInspector.DTYPE_NAMES.get(dtype, f"UNK({dtype})"),
                            "dtype_code": dtype,
                            "size_bytes": size_bytes,
                            "offset": offset
                        })
                        result["total_size"] += size_bytes
                    except:
                        break
                
                result["tensors"] = tensors
                result["valid"] = True
                
        except Exception as e:
            result["error"] = str(e)
        
        return result


# =============================================================================
# MAIN UI
# =============================================================================

class ConverterUI:
    """Main GUI window for GGUF Converter"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.configure(bg=COLORS['bg'])
        
        # UI Queue for async updates (prevents UI freezing)
        self.ui_queue = Queue()
        self.queue_processing = False
        
        # Load settings
        self.settings = load_settings()
        self.current_lang = tk.StringVar(value=self.settings.get("language", "ru"))
        
        self.root.title(self._t("window_title"))
        self._set_dark_titlebar()
        self.root.withdraw()
        self.root.geometry("720x728")
        self.root.resizable(False, False)
        
        # Set window icon
        if ICON_PATH.exists():
            try:
                self.root.iconbitmap(str(ICON_PATH))
                import ctypes
                myappid = 'gguf.converter.gui.v002'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except: pass
        
        # State
        self.model_path = tk.StringVar()
        self.quant_type = tk.StringVar(value="Q4_K_M")
        self.model_info: Optional[ModelInfo] = None
        self.converter: Optional[GGUFConverter] = None
        self.conversion_thread: Optional[threading.Thread] = None
        self.start_time = 0
        self.timer_running = False
        self.logo_image = None
        
        # Output folder
        self.output_folder = tk.StringVar(value=self.settings.get("output_folder", str(DOWNLOADS_FOLDER)))
        
        # Batch mode
        self.batch_mode = tk.BooleanVar(value=False)
        self.batch_files: List[Path] = []
        
        # Progress animation
        self.progress_value = 0
        self.progress_animating = False
        self.gradient_offset = 0.0
        self.wave_offset = 0.0
        self.is_converting = False
        
        # Animation thread for smooth independent animation
        self._animation_thread = None
        self._animation_running = False
        self._animation_lock = threading.Lock()
        self._last_animation_time = 0.0
        
        # Nyan Cat
        self.nyan_frames = []
        self.nyan_frame_index = 0
        self.nyan_width = 0
        self.nyan_height = 0
        
        # Sound
        self.sound_enabled = True
        self.wmp = None
        
        # Rainbow colors
        self.gradient_colors = [
            '#00ff87', '#00e5ff', '#00b8ff', '#0088ff', '#0066ff', '#4d5fff',
            '#8c52ff', '#b84dff', '#e052ff', '#ff4da6', '#ff5252', '#ff6b52',
            '#ff8c52', '#ffb852', '#ffe052', '#d4ff52', '#87ff52', '#52ff87',
        ]
        
        # Pre-created canvas items for efficient animation (no delete/create cycle)
        self._progress_segments = []  # List of rectangle item IDs
        self._nyan_item = None  # Nyan cat image item ID
        self._num_segments = 30  # Reduced segments for better performance
        
        self._create_ui()
        self._scan_downloads()
        self._load_nyan_cat()
        self.draw_progress_bar()
        self._center_window()
        self.root.deiconify()
        self._set_dark_titlebar()
        # Музыка запускается только при начале конвертации, не при старте приложения
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _t(self, key: str) -> str:
        """Get translated text."""
        return get_text(self.current_lang.get(), key)
    
    def _set_dark_titlebar(self):
        """Set dark title bar for Windows 11."""
        try:
            import ctypes
            from ctypes import wintypes
            self.root.update_idletasks()
            hwnd = self.root.winfo_id()
            GetAncestor = ctypes.windll.user32.GetAncestor
            GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
            GetAncestor.restype = wintypes.HWND
            root_hwnd = GetAncestor(hwnd, 2)
            if root_hwnd: hwnd = root_hwnd
            DwmSetWindowAttribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
            DwmSetWindowAttribute.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            dark_mode = ctypes.c_int(1)
            DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode))
            title_bar_color = ctypes.c_int(0x004b4b4b)
            DwmSetWindowAttribute(hwnd, 35, ctypes.byref(title_bar_color), ctypes.sizeof(title_bar_color))
            self.root.update()
        except: pass
    
    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
    
    def _on_close(self):
        self.stop_sound()
        self.root.destroy()
    
    def _load_nyan_cat(self):
        try:
            nyan_path = NYAN_PATH
            if not nyan_path.exists():
                nyan_path = SCRIPT_DIR / "nayan.gif"
            if nyan_path.exists() and PIL_AVAILABLE:
                nyan_gif = Image.open(nyan_path)
                target_height = 50
                aspect_ratio = nyan_gif.width / nyan_gif.height
                target_width = int(target_height * aspect_ratio)
                self.nyan_width = target_width
                self.nyan_height = target_height
                try:
                    while True:
                        frame = nyan_gif.copy()
                        frame = frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        self.nyan_frames.append(ImageTk.PhotoImage(frame))
                        nyan_gif.seek(len(self.nyan_frames))
                except EOFError: pass
        except: pass
    
    def _on_language_changed(self):
        """Handle language change."""
        lang = self.current_lang.get()
        self.settings["language"] = lang
        save_settings(self.settings)
        self._update_ui_texts()
    
    def _update_ui_texts(self):
        """Update all UI texts after language change."""
        self.root.title(self._t("window_title"))
        self.title_label.config(text=self._t("title"))
        self.version_label.config(text=self._t("version_text"))
        self.model_label.config(text=self._t("model_from_downloads"))
        self.quant_label.config(text=self._t("quantization_level"))
        self.status_text_label.config(text=self._t("status"))
        self.log_label.config(text=self._t("log"))
        self.status_label.config(text=self._t("ready"))
        # Update timer label prefix
        self.timer_label.config(text=f"{self._t('time')}: 00:00:00")
        # Update buttons
        self.convert_btn.config_text(self._t("convert"))
        self.cancel_btn.config_text(self._t("cancel"))
        self.refresh_btn.config_text(self._t("refresh"))
        self.browse_btn.config_text(self._t("browse"))
        # Update new elements
        self.output_folder_label.config(text=self._t("output_folder"))
        self.select_folder_btn.config_text(self._t("select_folder"))
        self.batch_check.config(text=self._t("batch_mode"))
        self.inspect_btn.config_text(self._t("inspect_gguf"))
        self._update_quant_desc()
        # Update model info label if model is selected
        if hasattr(self, 'model_info') and self.model_info:
            self._refresh_model_info_label()

    def _refresh_model_info_label(self):
        """Refresh model info label with current language."""
        if not self.model_info:
            return
        info = f"{self._t('type')}: {self.model_info.model_type}"
        info += f"  |  {self._t('size')}: {self.model_info.size_bytes / (1024*1024):.1f} MB"
        if self.model_info.dtype:
            info += f"  |  {self._t('dtype')}: {self.model_info.dtype}"
        if self.model_info.num_tensors:
            info += f"  |  {self._t('tensors')}: {self.model_info.num_tensors}"
        self.info_label.config(text=info)
        self._update_output_preview()

    def _select_output_folder(self):
        """Select output folder for converted files."""
        folder = filedialog.askdirectory(initialdir=self.output_folder.get())
        if folder:
            self.output_folder.set(folder)
            self.settings["output_folder"] = folder
            save_settings(self.settings)
            self._log(f"{self._t('output_folder')} {folder}")
    
    def _on_batch_mode_changed(self):
        """Handle batch mode toggle."""
        if self.batch_mode.get():
            self._select_batch_files()
        else:
            self.batch_files = []
            self.batch_files_label.config(text="")
    
    def _select_batch_files(self):
        """Select multiple files for batch conversion."""
        files = filedialog.askopenfilenames(
            initialdir=DOWNLOADS_FOLDER,
            filetypes=[("Model files", "*.safetensors *.pt *.pth *.bin"), ("All", "*.*")]
        )
        if files:
            self.batch_files = [Path(f) for f in files]
            self.batch_files_label.config(text=f"{self._t('files_selected')}: {len(self.batch_files)}")
            self._log(f"{self._t('files_selected')}: {len(self.batch_files)}")
        else:
            self.batch_mode.set(False)
            self.batch_files = []
            self.batch_files_label.config(text="")
    
    def _show_gguf_inspector(self):
        """Show GGUF file inspector dialog."""
        # Select GGUF file
        path = filedialog.askopenfilename(
            initialdir=DOWNLOADS_FOLDER,
            filetypes=[("GGUF files", "*.gguf"), ("All", "*.*")]
        )
        if not path:
            return
        
        path = Path(path)
        info = GGUFInspector.inspect(path)
        
        if not info["valid"]:
            show_dark_dialog(self.root, self._t("error"), info.get("error", self._t("invalid_gguf")), "error")
            return
        
        # Create inspector window
        inspector = tk.Toplevel(self.root)
        inspector.title(f"{self._t('inspect_title')} - {path.name}")
        inspector.configure(bg=COLORS['bg'])
        inspector.geometry("800x600")
        inspector.transient(self.root)
        
        # Apply dark titlebar
        try:
            import ctypes
            from ctypes import wintypes
            inspector.update_idletasks()
            hwnd = inspector.winfo_id()
            GetAncestor = ctypes.windll.user32.GetAncestor
            GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
            GetAncestor.restype = wintypes.HWND
            root_hwnd = GetAncestor(hwnd, 2)
            if root_hwnd: hwnd = root_hwnd
            DwmSetWindowAttribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
            DwmSetWindowAttribute.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            dark_mode = ctypes.c_int(1)
            DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode))
            title_bar_color = ctypes.c_int(0x004b4b4b)
            DwmSetWindowAttribute(hwnd, 35, ctypes.byref(title_bar_color), ctypes.sizeof(title_bar_color))
        except: pass
        
        main_frame = tk.Frame(inspector, bg=COLORS['bg'], padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header info
        header = tk.Frame(main_frame, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(header, text=path.name, font=("Segoe UI", 12, "bold"), fg=COLORS['fg'], bg=COLORS['bg']).pack(anchor=tk.W)
        
        info_text = f"{self._t('gguf_version')}: {info['version']}  |  "
        info_text += f"{self._t('total_tensors')}: {info['num_tensors']}  |  "
        info_text += f"{self._t('total_size')}: {info['total_size'] / (1024*1024):.1f} MB"
        tk.Label(header, text=info_text, font=("Segoe UI", 9), fg=COLORS['fg_dim'], bg=COLORS['bg']).pack(anchor=tk.W)
        
        # Tensor list with scrollbar
        list_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(list_frame, text=self._t("tensor_list"), font=("Segoe UI", 10), fg=COLORS['fg_dim'], bg=COLORS['bg']).pack(anchor=tk.W)
        
        # Create treeview for tensor list
        columns = ("name", "shape", "dtype", "size")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=20)
        
        tree.heading("name", text=self._t("tensor_name"))
        tree.heading("shape", text=self._t("tensor_shape"))
        tree.heading("dtype", text=self._t("tensor_dtype"))
        tree.heading("size", text=self._t("tensor_size"))
        
        tree.column("name", width=400)
        tree.column("shape", width=150)
        tree.column("dtype", width=80)
        tree.column("size", width=100)
        
        # Style treeview
        style = ttk.Style()
        style.configure("Treeview", background=COLORS['log_bg'], foreground=COLORS['fg'], 
                       fieldbackground=COLORS['log_bg'], font=("Consolas", 9))
        style.configure("Treeview.Heading", background=COLORS['button'], foreground=COLORS['fg'],
                       font=("Segoe UI", 9, "bold"))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate tensor list
        for tensor in info["tensors"]:
            shape_str = "×".join(str(d) for d in tensor["shape"])
            size_str = f"{tensor['size_bytes'] / 1024:.1f} KB" if tensor['size_bytes'] < 1024*1024 else f"{tensor['size_bytes'] / (1024*1024):.2f} MB"
            tree.insert("", tk.END, values=(tensor["name"], shape_str, tensor["dtype"], size_str))
        
        # Buttons
        btn_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        def copy_to_clipboard():
            text = f"File: {path.name}\n"
            text += f"Version: {info['version']}\n"
            text += f"Tensors: {info['num_tensors']}\n"
            text += f"Total size: {info['total_size'] / (1024*1024):.1f} MB\n\n"
            for t in info["tensors"]:
                text += f"{t['name']}: {t['shape']} {t['dtype']}\n"
            inspector.clipboard_clear()
            inspector.clipboard_append(text)
            self._log(f"{self._t('copy_info')}: OK")
        
        def export_csv():
            csv_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"{path.stem}_tensors.csv"
            )
            if csv_path:
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("name,shape,dtype,size_bytes\n")
                    for t in info["tensors"]:
                        shape_str = "×".join(str(d) for d in t["shape"])
                        f.write(f'"{t["name"]}","{shape_str}",{t["dtype"]},{t["size_bytes"]}\n')
                self._log(f"{self._t('export_csv')}: {csv_path}")
        
        copy_btn = make_round_button(btn_frame, self._t("copy_info"), copy_to_clipboard, bg_color=COLORS['bg'])
        copy_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = make_round_button(btn_frame, self._t("export_csv"), export_csv, bg_color=COLORS['bg'])
        export_btn.pack(side=tk.LEFT)
        
        close_btn = make_round_button(btn_frame, "OK", inspector.destroy, bg_color=COLORS['bg'])
        close_btn.pack(side=tk.RIGHT)

    
    def _create_ui(self):
        """Create all UI widgets."""
        main = tk.Frame(self.root, bg=COLORS['bg'], padx=20, pady=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        style = ttk.Style()
        style.theme_use('clam')
        # Separator style - darker than background
        style.configure('TSeparator', background='#2a2a2a')
        style.map('TSeparator', background=[('!disabled', '#2a2a2a')])
        
        # Combobox dark style - configure early before creating widgets
        style.configure("TCombobox", 
                       fieldbackground=COLORS['log_bg'],
                       background=COLORS['button'],
                       foreground=COLORS['fg'],
                       arrowcolor=COLORS['fg'],
                       bordercolor=COLORS['log_bg'],
                       lightcolor=COLORS['log_bg'],
                       darkcolor=COLORS['log_bg'],
                       insertcolor=COLORS['fg'],
                       borderwidth=0,
                       padding=(6, 5))  # Horizontal and vertical padding for slightly taller combobox
        style.map("TCombobox",
                 fieldbackground=[('readonly', COLORS['log_bg']), ('disabled', COLORS['bg'])],
                 background=[('readonly', COLORS['button']), ('active', COLORS['button_hover'])],
                 foreground=[('readonly', COLORS['fg']), ('disabled', COLORS['fg_dim'])],
                 bordercolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])],
                 lightcolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])],
                 darkcolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])],
                 selectbackground=[('readonly', COLORS['button'])],
                 selectforeground=[('readonly', COLORS['fg'])])
        
        # Entry dark style - same height as combobox
        style.configure("TEntry",
                       fieldbackground=COLORS['log_bg'],
                       background=COLORS['log_bg'],
                       foreground=COLORS['fg'],
                       bordercolor=COLORS['log_bg'],
                       lightcolor=COLORS['log_bg'],
                       darkcolor=COLORS['log_bg'],
                       insertcolor=COLORS['fg'],
                       borderwidth=0,
                       padding=(6, 5))
        style.map("TEntry",
                 fieldbackground=[('readonly', COLORS['log_bg']), ('disabled', COLORS['bg'])],
                 foreground=[('readonly', COLORS['fg']), ('disabled', COLORS['fg_dim'])],
                 bordercolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])],
                 lightcolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])],
                 darkcolor=[('focus', COLORS['log_bg']), ('!focus', COLORS['log_bg'])])
        
        # Style the combobox dropdown listbox (popup)
        self.root.option_add('*TCombobox*Listbox.background', COLORS['log_bg'])
        self.root.option_add('*TCombobox*Listbox.foreground', COLORS['fg'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', COLORS['button'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', COLORS['fg'])
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
        
        # Header
        header = tk.Frame(main, bg=COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 3))
        
        # Logo
        if LOGO_PATH.exists() and PIL_AVAILABLE:
            try:
                img = Image.open(LOGO_PATH)
                ratio = 70 / img.height
                new_size = (int(img.width * ratio), 70)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
                tk.Label(header, image=self.logo_image, bg=COLORS['bg']).pack(side=tk.LEFT, padx=(0, 15))
            except: pass
        
        # Title
        title_frame = tk.Frame(header, bg=COLORS['bg'])
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.title_label = tk.Label(title_frame, text=self._t("title"), font=("Segoe UI", 16, "bold"), fg=COLORS['fg'], bg=COLORS['bg'])
        self.title_label.pack(anchor=tk.W)
        
        # Version label (GPU indicator removed - CPU is faster for quantization)
        version_gpu_frame = tk.Frame(title_frame, bg=COLORS['bg'])
        version_gpu_frame.pack(anchor=tk.W)
        self.version_label = tk.Label(version_gpu_frame, text=self._t("version_text"), font=("Segoe UI", 8), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.version_label.pack(side=tk.LEFT)
        
        # CPU mode indicator (GPU not used - CPU is faster for quantization due to data transfer overhead)
        cpu_text = "  |  💻 CPU quantization (optimized)"
        self.gpu_label = tk.Label(version_gpu_frame, text=cpu_text, font=("Segoe UI", 8), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.gpu_label.pack(side=tk.LEFT)
        
        # Right side: Language selector only (sound button moved to bottom)
        header_right = tk.Frame(header, bg=COLORS['bg'])
        header_right.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Language selector with normal radio buttons (circles)
        lang_frame = tk.Frame(header_right, bg=COLORS['bg'])
        lang_frame.pack(side=tk.TOP)
        self.lang_buttons = {}
        for lang_code in ["ru", "en", "zh"]:
            lang_text = self._t(f"lang_{lang_code}")
            rb = tk.Radiobutton(lang_frame, text=lang_text, variable=self.current_lang, value=lang_code,
                font=("Segoe UI", 9), fg=COLORS['fg'], bg=COLORS['bg'], selectcolor=COLORS['bg'],
                activebackground=COLORS['bg'], activeforeground=COLORS['fg'], highlightthickness=0,
                command=self._on_language_changed)
            rb.pack(side=tk.LEFT, padx=3)
            self.lang_buttons[lang_code] = rb
        
        tk.Frame(main, height=1, bg='#2a2a2a').pack(fill=tk.X, pady=4)
        
        # Model selection
        self.model_label = tk.Label(main, text=self._t("model_from_downloads"), font=("Segoe UI", 10), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.model_label.pack(anchor=tk.W)
        
        combo_frame = tk.Frame(main, bg=COLORS['bg'])
        combo_frame.pack(fill=tk.X, pady=(2, 0))
        self.model_combo = ttk.Combobox(combo_frame, textvariable=self.model_path, state="readonly", font=("Segoe UI", 10))
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        btn_frame = tk.Frame(combo_frame, bg=COLORS['bg'])
        btn_frame.pack(side=tk.RIGHT, padx=(10, 0))
        self.refresh_btn = make_round_button(btn_frame, self._t("refresh"), self._scan_downloads, bg_color=COLORS['bg'])
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.browse_btn = make_round_button(btn_frame, self._t("browse"), self._browse_file, bg_color=COLORS['bg'])
        self.browse_btn.pack(side=tk.LEFT)
        
        self.info_label = tk.Label(main, text=" ", font=("Segoe UI", 9), fg=COLORS['fg_dim'], bg=COLORS['bg'], anchor=tk.W, height=1)
        self.info_label.pack(fill=tk.X, pady=(4, 0))
        self.output_label = tk.Label(main, text=" ", font=("Segoe UI", 9), fg=COLORS['accent'], bg=COLORS['bg'], anchor=tk.W, height=1)
        self.output_label.pack(fill=tk.X, pady=(2, 0))
        
        tk.Frame(main, height=1, bg='#2a2a2a').pack(fill=tk.X, pady=(6, 4))
        
        # Output folder selection
        output_frame = tk.Frame(main, bg=COLORS['bg'])
        output_frame.pack(fill=tk.X, pady=(2, 0))
        self.output_folder_label = tk.Label(output_frame, text=self._t("output_folder"), font=("Segoe UI", 10), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.output_folder_label.pack(side=tk.LEFT)
        
        # Use ttk.Entry styled like combobox for consistent height
        self.output_folder_entry = ttk.Entry(output_frame, textvariable=self.output_folder, font=("Segoe UI", 10))
        self.output_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.select_folder_btn = make_round_button(output_frame, self._t("select_folder"), self._select_output_folder, bg_color=COLORS['bg'], min_width=120)
        self.select_folder_btn.pack(side=tk.RIGHT)
        
        # Batch mode and Inspect buttons
        mode_frame = tk.Frame(main, bg=COLORS['bg'])
        mode_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.batch_check = tk.Checkbutton(mode_frame, text=self._t("batch_mode"), variable=self.batch_mode,
                                          font=("Segoe UI", 9), fg=COLORS['fg'], bg=COLORS['bg'],
                                          selectcolor=COLORS['button'], activebackground=COLORS['bg'],
                                          command=self._on_batch_mode_changed)
        self.batch_check.pack(side=tk.LEFT)
        
        self.batch_files_label = tk.Label(mode_frame, text="", font=("Segoe UI", 9), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.batch_files_label.pack(side=tk.LEFT, padx=(10, 0))
        
        self.inspect_btn = make_round_button(mode_frame, self._t("inspect_gguf"), self._show_gguf_inspector, bg_color=COLORS['bg'], min_width=180)
        self.inspect_btn.pack(side=tk.RIGHT)
        
        # Quantization
        self.quant_label = tk.Label(main, text=self._t("quantization_level"), font=("Segoe UI", 10), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.quant_label.pack(anchor=tk.W)
        
        quant_frame = tk.Frame(main, bg=COLORS['bg'])
        quant_frame.pack(fill=tk.X, pady=(0, 3))
        left_col = tk.Frame(quant_frame, bg=COLORS['bg'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_col = tk.Frame(quant_frame, bg=COLORS['bg'])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        for i, qt in enumerate(QuantType):
            col = left_col if i < 4 else right_col
            tk.Radiobutton(col, text=f"{qt.code} - {qt.label}", variable=self.quant_type, value=qt.code,
                font=("Segoe UI", 9), fg=COLORS['fg'], bg=COLORS['bg'], selectcolor=COLORS['button'],
                activebackground=COLORS['bg'], highlightthickness=0).pack(anchor=tk.W)
        
        self.quant_desc = tk.Label(main, text="", font=("Segoe UI", 9), fg=COLORS['fg_dim'], bg=COLORS['bg'])
        self.quant_desc.pack(anchor=tk.W)
        self.quant_type.trace_add("write", self._update_quant_desc)
        self._update_quant_desc()
        
        tk.Frame(main, height=1, bg='#2a2a2a').pack(fill=tk.X, pady=4)
        
        # Status
        status_frame = tk.Frame(main, bg=COLORS['bg'])
        status_frame.pack(fill=tk.X, pady=(2, 10))
        self.status_text_label = tk.Label(status_frame, text=self._t("status"), font=("Segoe UI", 9), fg='#999999', bg=COLORS['bg'])
        self.status_text_label.pack(side=tk.LEFT)
        self.status_label = tk.Label(status_frame, text=self._t("ready"), font=("Segoe UI", 9), fg=COLORS['accent'], bg=COLORS['bg'])
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Progress bar - full width layout with cat on left, percentage on right
        progress_container = tk.Frame(main, bg=COLORS['bg'])
        progress_container.pack(fill=tk.X, pady=2, padx=15)
        
        # Canvas takes most of the space
        self.progress_canvas = tk.Canvas(progress_container, width=600, height=60, bg=COLORS['bg'], highlightthickness=0)
        self.progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Percentage label on the right edge
        self.progress_label = tk.Label(progress_container, text="0%", font=("Segoe UI", 9), fg=COLORS['fg'], bg=COLORS['bg'], width=5, anchor='e')
        self.progress_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Pre-create canvas items for efficient animation (avoid delete/create cycle)
        self._init_progress_bar_items()
        
        # Log window (no scrollbar, mouse wheel scroll, reduced height)
        log_frame = tk.Frame(main, bg=COLORS['bg'])
        log_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.log_label = tk.Label(log_frame, text=self._t("log"), font=("Segoe UI", 9), fg='#999999', bg=COLORS['bg'])
        self.log_label.pack(anchor=tk.W)
        
        self.log_text = tk.Text(log_frame, height=4, font=("Consolas", 9), bg=COLORS['log_bg'], fg=COLORS['fg'],
                                wrap=tk.WORD, padx=10, pady=8, relief='flat', bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Mouse wheel scroll
        def on_mousewheel(event):
            self.log_text.yview_scroll(int(-1*(event.delta/120)), "units")
        self.log_text.bind("<MouseWheel>", on_mousewheel)
        
        # Timer
        timer_row = tk.Frame(log_frame, bg=COLORS['bg'])
        timer_row.pack(fill=tk.X, pady=(2, 0))
        self.timer_label = tk.Label(timer_row, text=f"{self._t('time')}: 00:00:00", font=("Segoe UI", 9), fg='#999999', bg=COLORS['bg'])
        self.timer_label.pack(side=tk.LEFT)
        
        # Buttons
        btn_row = tk.Frame(main, bg=COLORS['bg'])
        btn_row.pack(fill=tk.X, pady=(3, 5))
        # min_width=130 to fit "Конвертировать" (longest text)
        self.convert_btn = make_round_button(btn_row, self._t("convert"), self._start_conversion, bg_color=COLORS['bg'], min_width=130)
        self.convert_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.cancel_btn = make_round_button(btn_row, self._t("cancel"), self._cancel_conversion, bg_color=COLORS['bg'], min_width=90)
        self.cancel_btn.pack(side=tk.LEFT)
        self.cancel_btn.config(state='disabled')
        
        # Sound button (right side of button row)
        self.sound_btn = make_icon_button(btn_row, "🔊", self.toggle_sound, size=40, bg_color=COLORS['bg'])
        self.sound_btn.pack(side=tk.RIGHT)

    
    # ==================== Sound Methods ====================
    
    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_btn.config_icon("🔊")
            self.start_sound()
        else:
            self.sound_btn.config_icon("🔇")
            self.stop_sound()
    
    def start_sound(self):
        if not self.sound_enabled or self.wmp is not None:
            return
        try:
            import win32com.client
            if not MUSIC_FOLDER.exists():
                MUSIC_FOLDER.mkdir(parents=True)
                return
            self._music_files = list(MUSIC_FOLDER.glob("*.mp3")) + list(MUSIC_FOLDER.glob("*.wav"))
            if not self._music_files:
                self._log(self._t("no_music_files"))
                return
            self._play_random_track()
        except ImportError:
            self.wmp = None
        except Exception as e:
            self._log(f"{self._t('music_error')}: {e}")
            self.wmp = None
    
    def _play_random_track(self):
        """Play a random track from the music folder."""
        if not self.sound_enabled or not hasattr(self, '_music_files') or not self._music_files:
            return
        try:
            import win32com.client
            # Stop current playback if any
            if self.wmp is not None:
                try:
                    self.wmp.controls.stop()
                    self.wmp.close()
                except:
                    pass
            
            sound_path = random.choice(self._music_files)
            self._log(f"{self._t('playing')}: {sound_path.name}")
            self.wmp = win32com.client.Dispatch("WMPlayer.OCX")
            self.wmp.settings.setMode("loop", False)  # No loop - we'll handle track change
            self.wmp.settings.volume = 50
            self.wmp.URL = str(sound_path.absolute())
            self.wmp.controls.play()
            
            # Start monitoring for track end
            self._start_track_monitor()
        except Exception as e:
            self._log(f"{self._t('music_error')}: {e}")
            self.wmp = None
    
    def _start_track_monitor(self):
        """Monitor playback status and play next random track when current ends."""
        if not self.sound_enabled or self.wmp is None:
            return
        try:
            # WMP PlayState: 1=Stopped, 2=Paused, 3=Playing, 8=MediaEnded
            state = self.wmp.playState
            if state == 1 or state == 8:  # Stopped or MediaEnded
                # Track ended, play next random track
                self._play_random_track()
            elif state == 3:  # Still playing
                # Check again in 1 second
                self.root.after(1000, self._start_track_monitor)
        except:
            pass  # WMP might be closed
    
    def stop_sound(self):
        if self.wmp is not None:
            try:
                self.wmp.controls.stop()
                self.wmp.close()
            except: pass
            self.wmp = None
    
    # ==================== Progress Bar ====================
    
    def _init_progress_bar_items(self):
        """Pre-create canvas items for efficient animation (no delete/create cycle)."""
        # Create segments (initially hidden at 0,0,0,0)
        self._progress_segments = []
        for _ in range(self._num_segments):
            item_id = self.progress_canvas.create_rectangle(0, 0, 0, 0, fill='', outline='', state='hidden')
            self._progress_segments.append(item_id)
        
        # Create nyan cat image placeholder (will be configured when frames are loaded)
        self._nyan_item = self.progress_canvas.create_image(0, 30, image=None, anchor=tk.W, state='hidden')
    
    def draw_progress_bar(self):
        """Update progress bar using pre-created items (no delete/create - much faster)."""
        canvas_width = self.progress_canvas.winfo_width()
        if canvas_width < 100:  # Not yet rendered
            canvas_width = 600
        width = canvas_width - 20  # Leave small margin
        height = 22
        canvas_height = 60
        y_offset = (canvas_height - height) // 2
        progress_width = int((self.progress_value / 100) * width) - 25
        
        # Read animation state with lock for thread safety
        with self._animation_lock:
            gradient_offset = self.gradient_offset
            wave_offset = self.wave_offset
            nyan_frame_index = self.nyan_frame_index
        
        if progress_width > 0:
            segment_width = progress_width / self._num_segments
            
            # Calculate wave phase so that the END of the bar matches nyan cat position
            # Nyan cat uses: 8 * sin(wave_offset)
            # Last segment should have same phase, so we offset from the end
            for i, item_id in enumerate(self._progress_segments):
                x1 = int(i * segment_width)
                x2 = int((i + 1) * segment_width)
                if x2 > progress_width: x2 = progress_width
                
                # Smooth color calculation with interpolation
                color_pos = (i / self._num_segments * len(self.gradient_colors) + gradient_offset) % len(self.gradient_colors)
                color_idx = int(color_pos)
                color_frac = color_pos - color_idx
                next_idx = (color_idx + 1) % len(self.gradient_colors)
                color = self._interpolate_color(self.gradient_colors[color_idx], self.gradient_colors[next_idx], color_frac)
                
                # Wave effect - calculate from the END so it syncs with nyan cat
                # i=last should have phase = wave_offset (same as nyan)
                segments_from_end = self._num_segments - 1 - i
                wave_height = 5 * math.sin(wave_offset - segments_from_end / 3)
                
                center_y = y_offset + height // 2
                y1 = int(center_y - height // 2 + wave_height)
                y2 = int(center_y + height // 2 + wave_height)
                
                # Update existing item instead of delete/create
                self.progress_canvas.coords(item_id, x1, y1, x2, y2)
                self.progress_canvas.itemconfig(item_id, fill=color, state='normal')
        else:
            # Hide all segments when no progress
            for item_id in self._progress_segments:
                self.progress_canvas.itemconfig(item_id, state='hidden')
        
        # Update nyan cat position
        if self.nyan_frames and self._nyan_item:
            if progress_width > 0:
                nyan_x = min(progress_width - 10, width - 8)
                nyan_wave = 5 * math.sin(wave_offset)  # Same amplitude as bar end
            else:
                nyan_x = 0
                nyan_wave = 3 * math.sin(wave_offset)
            
            nyan_y = canvas_height // 2 + int(nyan_wave)
            current_frame = self.nyan_frames[nyan_frame_index]
            
            # Update existing item instead of delete/create
            self.progress_canvas.coords(self._nyan_item, nyan_x, nyan_y)
            self.progress_canvas.itemconfig(self._nyan_item, image=current_frame, state='normal')
            self.progress_canvas.tag_raise(self._nyan_item)
    
    def _interpolate_color(self, color1: str, color2: str, t: float) -> str:
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def start_progress_animation(self):
        if not self.progress_animating:
            self.progress_animating = True
            self._animation_running = True
            self._nyan_anim_counter = 0
            self._last_animation_time = time.perf_counter()
            
            # Start animation thread
            self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._animation_thread.start()
            
            # Start UI update loop (separate from animation calculations)
            self._schedule_animation_draw()
    
    def stop_progress_animation(self):
        self.progress_animating = False
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=0.2)
            self._animation_thread = None
    
    def _animation_loop(self):
        """Background thread for animation calculations - runs independently of UI load."""
        target_fps = 20  # 20 FPS for smooth animation (matches UI update rate)
        frame_time = 1.0 / target_fps
        
        while self._animation_running:
            start = time.perf_counter()
            
            # Update animation state (thread-safe)
            with self._animation_lock:
                self.gradient_offset = (self.gradient_offset + 0.18) % len(self.gradient_colors)
                self.wave_offset = (self.wave_offset + 0.3) % (2 * math.pi)
                
                # Update nyan frame every ~100ms (every 2nd frame at 20fps)
                if self.nyan_frames:
                    self._nyan_anim_counter += 1
                    if self._nyan_anim_counter >= 2:
                        self._nyan_anim_counter = 0
                        self.nyan_frame_index = (self.nyan_frame_index + 1) % len(self.nyan_frames)
            
            # Sleep to maintain target FPS
            elapsed = time.perf_counter() - start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _schedule_animation_draw(self):
        """Schedule next animation draw on main thread."""
        if self.progress_animating and self.is_converting:
            self.draw_progress_bar()
            # Use 50ms interval (20 FPS) for UI updates - sufficient for smooth animation
            self.root.after(50, self._schedule_animation_draw)
    
    def _animate_progress(self):
        """Legacy method - kept for compatibility but now uses threaded animation."""
        pass  # Animation is now handled by _animation_loop and _schedule_animation_draw
    
    # ==================== UI Methods ====================
    
    def _scan_downloads(self):
        models = []
        if DOWNLOADS_FOLDER.exists():
            for ext in [".safetensors", ".pt", ".pth", ".bin"]:
                for f in DOWNLOADS_FOLDER.glob(f"*{ext}"):
                    size = f.stat().st_size / (1024 * 1024)
                    models.append(f"{f.name} ({size:.0f} MB)")
        self.model_combo['values'] = sorted(models)
        if models:
            self.model_combo.current(0)
            self._on_model_selected(None)
        self._log(f"{self._t('found_models')}: {len(models)}")
    
    def _browse_file(self):
        path = filedialog.askopenfilename(initialdir=DOWNLOADS_FOLDER,
            filetypes=[("Model files", "*.safetensors *.pt *.pth *.bin"), ("All", "*.*")])
        if path:
            self.model_path.set(Path(path).name)
            self._analyze_model(Path(path))
    
    def _on_model_selected(self, event):
        sel = self.model_path.get()
        if sel:
            name = sel.split(" (")[0]
            self._analyze_model(DOWNLOADS_FOLDER / name)
    
    def _analyze_model(self, path: Path):
        log_to_file(f"_analyze_model called with path: {path}")
        try:
            self._log(f"{self._t('analyzing')}: {path.name}")
            self.model_info = ModelAnalyzer.analyze(path)
            log_to_file(f"model_info: type={self.model_info.model_type}, size={self.model_info.size_bytes}, dtype={self.model_info.dtype}, tensors={self.model_info.num_tensors}, model_name={self.model_info.model_name}")
            
            if self.model_info.model_name:
                config_name = _extract_name_from_config(path)
                if config_name and self.model_info.model_name == config_name:
                    self._log(f"{self._t('model_name_from_config')}: {self.model_info.model_name}")
                elif self.model_info.model_name != path.stem and self.model_info.model_name != path.parent.name:
                    self._log(f"{self._t('model_name_from_metadata')}: {self.model_info.model_name}")
                elif self.model_info.model_name == path.parent.name:
                    self._log(f"{self._t('model_name_from_folder')}: {self.model_info.model_name}")
                else:
                    self._log(f"{self._t('model_name_from_filename')}: {self.model_info.model_name}")
            
            info = f"{self._t('type')}: {self.model_info.model_type}"
            info += f"  |  {self._t('size')}: {self.model_info.size_bytes / (1024*1024):.1f} MB"
            if self.model_info.dtype:
                info += f"  |  {self._t('dtype')}: {self.model_info.dtype}"
            if self.model_info.num_tensors:
                info += f"  |  {self._t('tensors')}: {self.model_info.num_tensors}"
            log_to_file(f"Setting info_label text: {info}")
            self.info_label.config(text=info)
            self._update_output_preview()
        except Exception as e:
            log_exception(f"Error in _analyze_model: {e}")
    
    def _update_output_preview(self):
        if self.model_info:
            qtype = self.quant_type.get()
            out_name = f"{self.model_info.model_name}-{qtype}.gguf"
            self.output_label.config(text=f"{self._t('output')}: {out_name}")
    
    def _update_quant_desc(self, *args):
        code = self.quant_type.get()
        for qt in QuantType:
            if qt.code == code:
                self.quant_desc.config(text=qt.description)
                break
        self._update_output_preview()
    
    def _log(self, msg):
        """Log message to UI. Direct update when queue not active, queued during conversion."""
        if self.queue_processing:
            # During conversion, use queue for async updates
            self.ui_queue.put(('log', msg))
        else:
            # Outside conversion, update directly
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
    
    def _update_progress(self, val, status):
        """Queue progress update for async UI update."""
        # Only queue the value update, don't trigger redraw - animation handles that
        self.ui_queue.put(('progress', val, status))
    
    def _process_ui_queue(self):
        """Process queued UI updates (runs on main thread)."""
        if not self.queue_processing:
            return
        
        # Process ALL items in queue to prevent buildup
        while True:
            try:
                item = self.ui_queue.get_nowait()
                if item[0] == 'log':
                    msg = item[1]
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                elif item[0] == 'progress':
                    val, status = item[1], item[2]
                    self.progress_value = val
                    self.progress_label.config(text=f"{val}%")
                    self.status_label.config(text=status)
                    # DON'T call draw_progress_bar() here - animation loop handles it
            except Empty:
                break
        
        # Schedule next queue processing (100ms is enough for UI responsiveness)
        if self.queue_processing:
            self.root.after(100, self._process_ui_queue)
    
    def _start_queue_processing(self):
        """Start processing UI queue."""
        if not self.queue_processing:
            self.queue_processing = True
            self._process_ui_queue()
    
    def _stop_queue_processing(self):
        """Stop processing UI queue and flush remaining items."""
        self.queue_processing = False
        # Flush remaining items
        while True:
            try:
                item = self.ui_queue.get_nowait()
                if item[0] == 'log':
                    msg = item[1]
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                elif item[0] == 'progress':
                    val, status = item[1], item[2]
                    self.progress_value = val
                    self.progress_label.config(text=f"{val}%")
                    self.status_label.config(text=status)
                    # Final redraw after all items processed
            except Empty:
                break
        # One final redraw after flushing
        self.draw_progress_bar()
    
    def _update_timer(self):
        if self.timer_running and self.start_time:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            self.timer_label.config(text=f"{self._t('time')}: {hours:02d}:{mins:02d}:{secs:02d}")
            self.root.after(1000, self._update_timer)
    
    def _start_conversion(self):
        if self.batch_mode.get() and self.batch_files:
            self._start_batch_conversion()
            return
            
        if not self.model_info:
            show_dark_dialog(self.root, self._t("error"), self._t("select_model_first"), "error")
            return
        if self.model_info.error and "GGUF" in self.model_info.error:
            show_dark_dialog(self.root, self._t("info"), self._t("already_gguf"), "info")
            return
        
        qtype = None
        for qt in QuantType:
            if qt.code == self.quant_type.get():
                qtype = qt
                break
        
        self.convert_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.is_converting = True
        self.converter = GGUFConverter(progress_cb=self._update_progress, log_cb=self._log)
        self.start_time = time.time()
        self.timer_running = True
        self._update_timer()
        self.start_progress_animation()
        self._start_queue_processing()  # Start async UI updates
        self.start_sound()  # Запускаем музыку при начале конвертации
        
        # Get output folder
        out_dir = Path(self.output_folder.get())
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        
        self.conversion_thread = threading.Thread(target=self._run_conversion, args=(qtype, out_dir), daemon=True)
        self.conversion_thread.start()
    
    def _start_batch_conversion(self):
        """Start batch conversion of multiple files."""
        if not self.batch_files:
            show_dark_dialog(self.root, self._t("error"), self._t("select_model_first"), "error")
            return
        
        qtype = None
        for qt in QuantType:
            if qt.code == self.quant_type.get():
                qtype = qt
                break
        
        self.convert_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.is_converting = True
        self.start_time = time.time()
        self.timer_running = True
        self._update_timer()
        self.start_progress_animation()
        self._start_queue_processing()  # Start async UI updates
        self.start_sound()
        
        out_dir = Path(self.output_folder.get())
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        
        self.conversion_thread = threading.Thread(
            target=self._run_batch_conversion, 
            args=(qtype, out_dir), 
            daemon=True
        )
        self.conversion_thread.start()
    
    def _run_batch_conversion(self, qtype, out_dir):
        """Run batch conversion in background thread."""
        results = []
        total = len(self.batch_files)
        
        for i, file_path in enumerate(self.batch_files):
            if self.converter and self.converter._cancelled:
                break
            
            self._log(f"\n{'='*40}")
            self._log(f"{self._t('batch_progress')} {i+1}/{total}: {file_path.name}")
            
            # Analyze model
            info = ModelAnalyzer.analyze(file_path)
            if info.error and "GGUF" in info.error:
                results.append((file_path.name, False, "Already GGUF"))
                continue
            
            # Create converter for this file
            self.converter = GGUFConverter(
                progress_cb=lambda v, s: self._update_progress(int((i * 100 + v) / total), s),
                log_cb=self._log
            )
            
            result = self.converter.convert(info, qtype, out_dir)
            results.append((file_path.name, result.success, result.error_message or "OK"))
        
        self.root.after(0, lambda: self._show_batch_result(results))
    
    def _run_conversion(self, qtype, out_dir):
        try:
            result = self.converter.convert(self.model_info, qtype, out_dir)
            self.root.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._show_result(ConversionResult(False, error_message=str(e))))
    
    def _cancel_conversion(self):
        if self.converter:
            self.converter.cancel()
            self._log(self._t("cancelling"))
    
    def _show_result(self, result):
        self.is_converting = False
        self.timer_running = False
        self.stop_progress_animation()
        self._stop_queue_processing()  # Stop and flush UI queue
        self.stop_sound()  # Останавливаем музыку при завершении конвертации
        
        self.convert_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.model_combo.config(state="readonly")
        
        if result.success:
            self.progress_value = 100
            self.draw_progress_bar()
            self.status_label.config(text=self._t("conversion_complete"), fg=COLORS['success'])
            msg = f"{self._t('success')}\n\n{self._t('file')}: {result.output_path.name}\n"
            msg += f"{self._t('size')}: {result.input_size_mb:.1f} MB → {result.output_size_mb:.1f} MB\n"
            msg += f"{self._t('compression')}: {result.compression_ratio:.2f}x\n"
            msg += f"{self._t('time')}: {result.elapsed_time:.1f}s"
            show_dark_dialog(self.root, self._t("done"), msg, "info")
        else:
            self.progress_value = 0
            self.progress_label.config(text="0%")
            self.draw_progress_bar()
            self.status_label.config(text=self._t("conversion_failed"), fg=COLORS['error'])
            show_dark_dialog(self.root, self._t("error"), result.error_message or "Unknown error", "error")
        
        # Сбрасываем прогресс-бар и кота в начальное положение после закрытия диалога
        self.progress_value = 0
        self.progress_label.config(text="0%")
        self.draw_progress_bar()
    
    def _show_batch_result(self, results):
        """Show batch conversion results."""
        self.is_converting = False
        self.timer_running = False
        self.stop_progress_animation()
        self._stop_queue_processing()  # Stop and flush UI queue
        self.stop_sound()
        
        self.convert_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.model_combo.config(state="readonly")
        
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        self.progress_value = 100
        self.draw_progress_bar()
        self.status_label.config(text=self._t("batch_complete"), fg=COLORS['success'] if failed == 0 else COLORS['warning'])
        
        # Build result message
        msg = f"{self._t('batch_results')}:\n"
        msg += f"{self._t('successful')}: {successful}\n"
        msg += f"{self._t('failed')}: {failed}\n\n"
        
        for name, success, error in results:
            status = "✓" if success else "✗"
            msg += f"{status} {name}"
            if not success:
                msg += f" ({error})"
            msg += "\n"
        
        show_dark_dialog(self.root, self._t("batch_complete"), msg, "info" if failed == 0 else "warning")
        
        self.progress_value = 0
        self.progress_label.config(text="0%")
        self.draw_progress_bar()
    
    def run(self):
        self.root.mainloop()


def main():
    if not MUSIC_FOLDER.exists():
        MUSIC_FOLDER.mkdir(parents=True)
    
    app = ConverterUI()
    app.run()


if __name__ == "__main__":
    main()
