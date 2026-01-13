"""
GGUF Converter - UI Widgets
Reusable dark-themed UI components for tkinter

Author: miha2017
"""

import tkinter as tk
from tkinter import Canvas

# =============================================================================
# THEME AND COLORS
# =============================================================================

COLORS = {
    'bg': '#3a3a3a',
    'fg': '#e0e0e0',
    'fg_dim': '#959595',
    'accent': '#5a9fd4',
    'success': '#6ab04c',
    'warning': '#f0932b',
    'error': '#eb4d4b',
    'button': '#5a5a5a',
    'button_hover': '#6a6a6a',
    'button_active': '#4a4a4a',
    'button_disabled': '#3a3a3a',
    'border': '#777777',
    'border_hover': '#888888',
    'shadow': '#2a2a2a',
    'log_bg': '#2a2a2a',
}

BUTTON_RADIUS = 4
BUTTON_PADDING_X = 12
BUTTON_HEIGHT = 40

# =============================================================================
# ROUNDED BUTTONS
# =============================================================================

def make_round_button(parent, text, command, bg_color=None, min_width=None):
    """Create a rounded button with shadow and border."""
    if bg_color is None:
        bg_color = COLORS['bg']
    
    temp = Canvas(parent, bg=bg_color, highlightthickness=0)
    tid = temp.create_text(0, 0, text=text, font=('Arial', 10, 'bold'))
    bbox = temp.bbox(tid)
    text_width = bbox[2] - bbox[0] if bbox else len(text) * 8
    temp.destroy()
    
    btn_width = int(text_width + BUTTON_PADDING_X * 2 + 8)
    if min_width and btn_width < min_width:
        btn_width = min_width
    btn_height = BUTTON_HEIGHT
    
    canvas = Canvas(parent, width=btn_width, height=btn_height, bg=bg_color, highlightthickness=0, bd=0)
    canvas.button_text = text
    canvas.button_state = 'normal'

    def draw_button(color=None, text_color=None, shadow=True, border_color=None):
        if color is None: color = COLORS['button']
        if text_color is None: text_color = '#ffffff'
        if border_color is None: border_color = COLORS['border']
        
        canvas.delete('all')
        r = BUTTON_RADIUS
        
        if shadow:
            sx1, sy1 = 4, 4
            sx2, sy2 = btn_width - 2, btn_height - 2
            canvas.create_arc(sx1, sy1, sx1+r*2, sy1+r*2, start=90, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx2-r*2, sy1, sx2, sy1+r*2, start=0, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx1, sy2-r*2, sx1+r*2, sy2, start=180, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx2-r*2, sy2-r*2, sx2, sy2, start=270, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_rectangle(sx1+r, sy1, sx2-r, sy2, fill=COLORS['shadow'], outline='')
            canvas.create_rectangle(sx1, sy1+r, sx2, sy2-r, fill=COLORS['shadow'], outline='')
        
        x1, y1 = 2, 2
        x2, y2 = btn_width - 4, btn_height - 4
        canvas.create_arc(x1, y1, x1+r*2, y1+r*2, start=90, extent=90, fill=color, outline='')
        canvas.create_arc(x2-r*2, y1, x2, y1+r*2, start=0, extent=90, fill=color, outline='')
        canvas.create_arc(x1, y2-r*2, x1+r*2, y2, start=180, extent=90, fill=color, outline='')
        canvas.create_arc(x2-r*2, y2-r*2, x2, y2, start=270, extent=90, fill=color, outline='')
        canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline='')
        canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline='')
        
        canvas.create_arc(x1, y1, x1+r*2, y1+r*2, start=90, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x2-r*2, y1, x2, y1+r*2, start=0, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x1, y2-r*2, x1+r*2, y2, start=180, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x2-r*2, y2-r*2, x2, y2, start=270, extent=90, outline=border_color, style='arc')
        canvas.create_line(x1+r, y1, x2-r, y1, fill=border_color)
        canvas.create_line(x1+r, y2, x2-r, y2, fill=border_color)
        canvas.create_line(x1, y1+r, x1, y2-r, fill=border_color)
        canvas.create_line(x2, y1+r, x2, y2-r, fill=border_color)
        
        text_y = btn_height//2 - 2
        canvas.create_text(btn_width//2, text_y, text=canvas.button_text, fill=text_color, font=('Arial', 10, 'bold'))
    
    draw_button()
    
    def on_enter(e):
        if canvas.button_state == 'normal':
            draw_button(COLORS['button_hover'], '#ffffff', True, COLORS['border_hover'])
            canvas.config(cursor='hand2')
    
    def on_leave(e):
        if canvas.button_state == 'normal':
            draw_button()
            canvas.config(cursor='')
    
    def on_click(e):
        if canvas.button_state == 'normal':
            draw_button(COLORS['button_active'], '#ffffff', False)
            canvas.after(100, lambda: draw_button(COLORS['button_hover']) if canvas.button_state == 'normal' else None)
            if command: command()
    
    canvas.bind('<Enter>', on_enter)
    canvas.bind('<Leave>', on_leave)
    canvas.bind('<Button-1>', on_click)
    
    original_config = canvas.config
    def configure(**kwargs):
        if 'state' in kwargs:
            canvas.button_state = kwargs['state']
            if kwargs['state'] == 'disabled':
                canvas.unbind('<Enter>')
                canvas.unbind('<Leave>')
                canvas.unbind('<Button-1>')
                draw_button(COLORS['button_disabled'], '#666666', False, '#555555')
                canvas.config(cursor='')
            else:
                draw_button()
                canvas.bind('<Enter>', on_enter)
                canvas.bind('<Leave>', on_leave)
                canvas.bind('<Button-1>', on_click)
            kwargs = {k: v for k, v in kwargs.items() if k != 'state'}
        if kwargs: original_config(**kwargs)
    
    def update_text(new_text):
        canvas.button_text = new_text
        if canvas.button_state == 'disabled':
            draw_button(COLORS['button_disabled'], '#666666', False, '#555555')
        else:
            draw_button()
    
    canvas.config = configure
    canvas.configure = configure
    canvas.config_text = update_text
    return canvas


def make_icon_button(parent, icon, command, size=40, bg_color=None):
    """Create a square rounded button with icon (emoji)."""
    if bg_color is None:
        bg_color = COLORS['bg']
    
    canvas = Canvas(parent, width=size, height=size, bg=bg_color, highlightthickness=0, bd=0)
    canvas.button_icon = icon
    r = BUTTON_RADIUS
    
    def draw_button(color=None, icon_color=None, shadow=True, border_color=None):
        if color is None: color = COLORS['button']
        if icon_color is None: icon_color = '#cccccc'
        if border_color is None: border_color = COLORS['border']
        
        canvas.delete('all')
        
        if shadow:
            sx1, sy1 = 4, 4
            sx2, sy2 = size - 2, size - 2
            canvas.create_arc(sx1, sy1, sx1+r*2, sy1+r*2, start=90, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx2-r*2, sy1, sx2, sy1+r*2, start=0, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx1, sy2-r*2, sx1+r*2, sy2, start=180, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_arc(sx2-r*2, sy2-r*2, sx2, sy2, start=270, extent=90, fill=COLORS['shadow'], outline='')
            canvas.create_rectangle(sx1+r, sy1, sx2-r, sy2, fill=COLORS['shadow'], outline='')
            canvas.create_rectangle(sx1, sy1+r, sx2, sy2-r, fill=COLORS['shadow'], outline='')
        
        x1, y1 = 2, 2
        x2, y2 = size - 4, size - 4
        canvas.create_arc(x1, y1, x1+r*2, y1+r*2, start=90, extent=90, fill=color, outline='')
        canvas.create_arc(x2-r*2, y1, x2, y1+r*2, start=0, extent=90, fill=color, outline='')
        canvas.create_arc(x1, y2-r*2, x1+r*2, y2, start=180, extent=90, fill=color, outline='')
        canvas.create_arc(x2-r*2, y2-r*2, x2, y2, start=270, extent=90, fill=color, outline='')
        canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=color, outline='')
        canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=color, outline='')
        
        canvas.create_arc(x1, y1, x1+r*2, y1+r*2, start=90, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x2-r*2, y1, x2, y1+r*2, start=0, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x1, y2-r*2, x1+r*2, y2, start=180, extent=90, outline=border_color, style='arc')
        canvas.create_arc(x2-r*2, y2-r*2, x2, y2, start=270, extent=90, outline=border_color, style='arc')
        canvas.create_line(x1+r, y1, x2-r, y1, fill=border_color)
        canvas.create_line(x1+r, y2, x2-r, y2, fill=border_color)
        canvas.create_line(x1, y1+r, x1, y2-r, fill=border_color)
        canvas.create_line(x2, y1+r, x2, y2-r, fill=border_color)
        
        btn_center_x = size // 2
        btn_center_y = size // 2 - 6
        canvas.create_text(btn_center_x, btn_center_y, text=canvas.button_icon, fill=icon_color, font=('Segoe UI', 14))
    
    draw_button()
    
    def update_icon(new_icon):
        canvas.button_icon = new_icon
        draw_button()
    
    canvas.config_icon = update_icon
    
    def on_enter(e):
        draw_button(COLORS['button_hover'], '#ffffff', True, COLORS['border_hover'])
        canvas.config(cursor='hand2')
    
    def on_leave(e):
        draw_button()
        canvas.config(cursor='')
    
    def on_click(e):
        draw_button(COLORS['button_active'], '#ffffff', False)
        canvas.after(100, draw_button)
        if command: command()
    
    canvas.bind('<Enter>', on_enter)
    canvas.bind('<Leave>', on_leave)
    canvas.bind('<Button-1>', on_click)
    return canvas


# =============================================================================
# DARK DIALOG
# =============================================================================

def show_dark_dialog(parent, title: str, message: str, dialog_type: str = "info"):
    """Show a dark-themed dialog box."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=COLORS['bg'])
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()
    
    lines = message.count('\n') + 1
    max_line_len = max(len(line) for line in message.split('\n'))
    width = max(350, min(500, max_line_len * 8 + 80))
    height = max(200, 150 + lines * 22)
    
    dialog.geometry(f"{width}x{height}")
    dialog.update_idletasks()
    parent.update_idletasks()
    px = parent.winfo_x() + (parent.winfo_width() - width) // 2
    py = parent.winfo_y() + (parent.winfo_height() - height) // 2
    dialog.geometry(f"{width}x{height}+{px}+{py}")
    
    # Windows dark title bar
    try:
        import ctypes
        from ctypes import wintypes
        dialog.update_idletasks()
        hwnd = dialog.winfo_id()
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
    
    icons = {"info": ("✓", COLORS['success']), "error": ("✗", COLORS['error']), "warning": ("⚠", COLORS['warning'])}
    icon_char, icon_color = icons.get(dialog_type, icons["info"])
    
    content = tk.Frame(dialog, bg=COLORS['bg'])
    content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
    tk.Label(content, text=icon_char, font=("Segoe UI", 28), fg=icon_color, bg=COLORS['bg']).pack(pady=(5, 10))
    tk.Label(content, text=message, font=("Segoe UI", 10), fg=COLORS['fg'], bg=COLORS['bg'], justify=tk.LEFT, wraplength=width - 60).pack(pady=(0, 15))
    
    btn_frame = tk.Frame(content, bg=COLORS['bg'])
    btn_frame.pack(pady=(5, 0))
    ok_btn = make_round_button(btn_frame, "OK", dialog.destroy, bg_color=COLORS['bg'])
    ok_btn.pack()
    
    dialog.bind('<Return>', lambda e: dialog.destroy())
    dialog.bind('<Escape>', lambda e: dialog.destroy())
    dialog.focus_set()
    parent.wait_window(dialog)
