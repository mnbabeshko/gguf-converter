"""
Quick README preview - opens in browser
Usage: python preview_readme.py
"""
import webbrowser
import tempfile
import re
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Installing markdown...")
    import subprocess
    subprocess.run(["pip", "install", "markdown"], check=True)
    import markdown

# Read README
base_path = Path(__file__).parent
readme_path = base_path / "README_RU.md"  # или README.md
content = readme_path.read_text(encoding='utf-8')

# Replace relative image paths with absolute file:// URLs
def fix_image_paths(match):
    img_path = match.group(1)
    if not img_path.startswith(('http://', 'https://', 'file://')):
        abs_path = (base_path / img_path).resolve()
        return f'<img src="file:///{abs_path}"'
    return match.group(0)

# Convert to HTML
html = markdown.markdown(content, extensions=['tables', 'fenced_code'])

# Fix image paths in HTML
html = re.sub(r'<img src="([^"]+)"', fix_image_paths, html)

# GitHub-like CSS
css = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; 
       max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #24292e; }
h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; }
pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #dfe2e5; padding: 8px 12px; text-align: center; }
img { max-width: 150px; }
a { color: #0366d6; }
</style>
"""

full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{html}</body></html>"

# Save and open
with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as f:
    f.write(full_html)
    temp_path = f.name

webbrowser.open(f'file://{temp_path}')
print(f"Opened: {temp_path}")
