# GGUF Converter Launcher - PowerShell (no console flash)
# Run this script to start GGUF Converter without any console window

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Find pythonw.exe
$pythonPaths = @(
    "$env:USERPROFILE\ThePuppeteer\.venv\Scripts\pythonw.exe",
    "C:\Users\user\ThePuppeteer\.venv\Scripts\pythonw.exe",
    "$env:USERPROFILE\anaconda3\pythonw.exe",
    "$env:USERPROFILE\miniconda3\pythonw.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\pythonw.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\pythonw.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python310\pythonw.exe"
)

$pythonw = $null
foreach ($p in $pythonPaths) {
    if (Test-Path $p) {
        $pythonw = $p
        break
    }
}

# Fallback: try to find in PATH
if (-not $pythonw) {
    $pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
}

if (-not $pythonw) {
    [System.Windows.Forms.MessageBox]::Show("Python not found!", "GGUF Converter", "OK", "Error")
    exit 1
}

$script = Join-Path $scriptDir "gguf_converter.py"

# Start process completely hidden
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $pythonw
$psi.Arguments = "`"$script`""
$psi.WorkingDirectory = $scriptDir
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$psi.CreateNoWindow = $true
$psi.UseShellExecute = $false

[System.Diagnostics.Process]::Start($psi) | Out-Null
