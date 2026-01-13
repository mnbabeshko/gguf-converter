"""
Test script to measure progress bar animation performance.
Compares old (delete/create) vs new (coords/itemconfig) approach.
"""
import sys
import os
import time
import math
import tkinter as tk

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_old_approach():
    """Test old approach: delete and create items each frame."""
    root = tk.Tk()
    root.withdraw()
    
    canvas = tk.Canvas(root, width=620, height=60, bg='#3a3a3a', highlightthickness=0)
    canvas.pack()
    
    gradient_colors = [
        '#00ff87', '#00e5ff', '#00b8ff', '#0088ff', '#0066ff', '#4d5fff',
        '#8c52ff', '#b84dff', '#e052ff', '#ff4da6', '#ff5252', '#ff6b52',
        '#ff8c52', '#ffb852', '#ffe052', '#d4ff52', '#87ff52', '#52ff87',
    ]
    
    num_segments = 25
    num_frames = 100
    
    start = time.perf_counter()
    
    for frame in range(num_frames):
        # Delete all items
        canvas.delete("progress")
        
        progress_width = 500
        segment_width = progress_width / num_segments
        gradient_offset = frame * 0.12
        wave_offset = frame * 0.22
        
        for i in range(num_segments):
            x1 = int(i * segment_width)
            x2 = int((i + 1) * segment_width)
            
            color_idx = int((i / num_segments * len(gradient_colors) + gradient_offset)) % len(gradient_colors)
            color = gradient_colors[color_idx]
            
            wave_height = 5 * math.sin((i / 3) + wave_offset)
            y1 = int(19 - 11 + wave_height)
            y2 = int(19 + 11 + wave_height)
            
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="progress")
        
        root.update_idletasks()
    
    elapsed = time.perf_counter() - start
    root.destroy()
    return elapsed

def test_new_approach():
    """Test new approach: pre-create items and update with coords/itemconfig."""
    root = tk.Tk()
    root.withdraw()
    
    canvas = tk.Canvas(root, width=620, height=60, bg='#3a3a3a', highlightthickness=0)
    canvas.pack()
    
    gradient_colors = [
        '#00ff87', '#00e5ff', '#00b8ff', '#0088ff', '#0066ff', '#4d5fff',
        '#8c52ff', '#b84dff', '#e052ff', '#ff4da6', '#ff5252', '#ff6b52',
        '#ff8c52', '#ffb852', '#ffe052', '#d4ff52', '#87ff52', '#52ff87',
    ]
    
    num_segments = 25
    num_frames = 100
    
    # Pre-create items
    segments = []
    for _ in range(num_segments):
        item_id = canvas.create_rectangle(0, 0, 0, 0, fill='', outline='', state='hidden')
        segments.append(item_id)
    
    start = time.perf_counter()
    
    for frame in range(num_frames):
        progress_width = 500
        segment_width = progress_width / num_segments
        gradient_offset = frame * 0.12
        wave_offset = frame * 0.22
        
        for i, item_id in enumerate(segments):
            x1 = int(i * segment_width)
            x2 = int((i + 1) * segment_width)
            
            color_idx = int((i / num_segments * len(gradient_colors) + gradient_offset)) % len(gradient_colors)
            color = gradient_colors[color_idx]
            
            wave_height = 5 * math.sin((i / 3) + wave_offset)
            y1 = int(19 - 11 + wave_height)
            y2 = int(19 + 11 + wave_height)
            
            canvas.coords(item_id, x1, y1, x2, y2)
            canvas.itemconfig(item_id, fill=color, state='normal')
        
        root.update_idletasks()
    
    elapsed = time.perf_counter() - start
    root.destroy()
    return elapsed

if __name__ == "__main__":
    print("Progress Bar Animation Performance Test")
    print("=" * 50)
    print(f"Testing 100 frames with 25 segments each...")
    print()
    
    # Run tests multiple times for accuracy
    old_times = []
    new_times = []
    
    for i in range(3):
        print(f"Run {i+1}/3...")
        old_times.append(test_old_approach())
        new_times.append(test_new_approach())
    
    old_avg = sum(old_times) / len(old_times)
    new_avg = sum(new_times) / len(new_times)
    
    print()
    print("Results:")
    print("-" * 50)
    print(f"Old approach (delete/create): {old_avg*1000:.1f}ms for 100 frames")
    print(f"New approach (coords/config): {new_avg*1000:.1f}ms for 100 frames")
    print(f"Speedup: {old_avg/new_avg:.2f}x")
    print()
    print(f"Old: {old_avg*10:.2f}ms per frame ({1000/old_avg/10:.1f} FPS max)")
    print(f"New: {new_avg*10:.2f}ms per frame ({1000/new_avg/10:.1f} FPS max)")
