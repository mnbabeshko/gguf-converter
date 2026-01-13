#!/usr/bin/env python3
"""
Test script for progress bar animation performance.
Simulates conversion progress to verify smooth animation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
import threading
import time

def test_animation():
    """Test progress bar animation without actual conversion."""
    from gguf_converter import ConverterUI
    
    # Create UI
    ui = ConverterUI()
    
    def simulate_conversion():
        """Simulate conversion progress."""
        time.sleep(1)  # Wait for UI to initialize
        
        # Start animation
        ui.is_converting = True
        ui.root.after(0, ui.start_progress_animation)
        ui.root.after(0, ui._start_queue_processing)
        
        # Simulate progress updates
        for i in range(101):
            ui._update_progress(i, f"Тест анимации: {i}%")
            time.sleep(0.1)  # 10 seconds total
        
        # Stop animation
        ui.is_converting = False
        ui.root.after(0, ui.stop_progress_animation)
        ui.root.after(0, ui._stop_queue_processing)
        ui.root.after(0, lambda: ui.status_label.config(text="Тест завершён!"))
    
    # Start simulation in background thread
    thread = threading.Thread(target=simulate_conversion, daemon=True)
    thread.start()
    
    # Run UI
    ui.run()

if __name__ == "__main__":
    print("Testing progress bar animation...")
    print("Watch for smooth rainbow gradient and wave effect.")
    print("The animation should NOT freeze or stutter.")
    test_animation()
