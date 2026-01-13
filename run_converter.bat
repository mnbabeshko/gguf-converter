@echo off
REM GGUF Converter Launcher - runs VBS script silently
REM Double-click this file or "GGUF Converter.vbs" to start

cd /d "%~dp0"

REM Use wscript (not cscript) to run VBS without any console
start "" /b wscript.exe //nologo "GGUF Converter.vbs"
exit
