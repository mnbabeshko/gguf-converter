' Creates a desktop shortcut for GGUF Converter
' Run this once to create a shortcut that launches without any console flash

Option Explicit

Dim WshShell, fso, scriptDir, desktopPath, shortcut, pythonw, iconPath

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
desktopPath = WshShell.SpecialFolders("Desktop")

' Find pythonw
pythonw = FindPythonW()

If pythonw = "" Then
    MsgBox "Python not found! Please install Python first.", vbCritical, "GGUF Converter"
    WScript.Quit 1
End If

' Create shortcut
Set shortcut = WshShell.CreateShortcut(desktopPath & "\GGUF Converter.lnk")
shortcut.TargetPath = pythonw
shortcut.Arguments = """" & scriptDir & "\gguf_converter.py"""
shortcut.WorkingDirectory = scriptDir
shortcut.WindowStyle = 7  ' Minimized (but pythonw doesn't show window anyway)
shortcut.Description = "GGUF Model Converter"

' Set icon if exists
iconPath = scriptDir & "\images\logoLLM.ico"
If fso.FileExists(iconPath) Then
    shortcut.IconLocation = iconPath
End If

shortcut.Save

MsgBox "Shortcut created on Desktop!" & vbCrLf & vbCrLf & _
       "Use 'GGUF Converter' shortcut to start the program without any console window.", _
       vbInformation, "GGUF Converter"

Function FindPythonW()
    Dim paths, p, userProfile
    
    userProfile = WshShell.ExpandEnvironmentStrings("%USERPROFILE%")
    
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
        "C:\Python310\pythonw.exe" _
    )
    
    For Each p In paths
        If fso.FileExists(p) Then
            FindPythonW = p
            Exit Function
        End If
    Next
    
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
