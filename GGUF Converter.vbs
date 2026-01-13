' GGUF Converter Launcher - No Console Window
' This script launches the converter without showing any console window

Option Explicit

Dim WshShell, fso, scriptDir, pythonPath, scriptPath, cmd

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get script directory
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Find pythonw.exe (GUI Python without console)
pythonPath = FindPythonW()

If pythonPath = "" Then
    MsgBox "Python not found! Please install Python or check your PATH.", vbCritical, "GGUF Converter"
    WScript.Quit 1
End If

' Build command - use .py file directly (pythonw handles it without console)
scriptPath = scriptDir & "\gguf_converter.py"

If Not fso.FileExists(scriptPath) Then
    MsgBox "gguf_converter.py not found in " & scriptDir, vbCritical, "GGUF Converter"
    WScript.Quit 1
End If

cmd = """" & pythonPath & """ """ & scriptPath & """"

' Run completely hidden (0 = vbHide)
WshShell.Run cmd, 0, False

Function FindPythonW()
    Dim paths, p, userProfile
    
    userProfile = WshShell.ExpandEnvironmentStrings("%USERPROFILE%")
    
    ' Priority list of pythonw.exe locations
    paths = Array( _
        userProfile & "\ThePuppeteer\.venv\Scripts\pythonw.exe", _
        "C:\Users\user\ThePuppeteer\.venv\Scripts\pythonw.exe", _
        userProfile & "\anaconda3\pythonw.exe", _
        userProfile & "\miniconda3\pythonw.exe", _
        userProfile & "\AppData\Local\Programs\Python\Python312\pythonw.exe", _
        userProfile & "\AppData\Local\Programs\Python\Python311\pythonw.exe", _
        userProfile & "\AppData\Local\Programs\Python\Python310\pythonw.exe", _
        userProfile & "\AppData\Local\Programs\Python\Python39\pythonw.exe", _
        "C:\Python312\pythonw.exe", _
        "C:\Python311\pythonw.exe", _
        "C:\Python310\pythonw.exe", _
        "C:\Python39\pythonw.exe" _
    )
    
    For Each p In paths
        If fso.FileExists(p) Then
            FindPythonW = p
            Exit Function
        End If
    Next
    
    ' Try to find pythonw in PATH using where command
    On Error Resume Next
    Dim exec, result
    Set exec = WshShell.Exec("where pythonw.exe")
    result = exec.StdOut.ReadLine()
    On Error GoTo 0
    
    If result <> "" And fso.FileExists(result) Then
        FindPythonW = result
        Exit Function
    End If
    
    FindPythonW = ""
End Function
