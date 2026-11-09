' Silent GUI launcher: no cmd.exe window.
Option Explicit
Dim shell, appDir, gui, candidate, launched
Set shell = CreateObject("WScript.Shell")
appDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
gui = Chr(34) & appDir & "\gui.py" & Chr(34)
launched = False

candidate = appDir & "\.venv\Scripts\pythonw.exe"
If CreateObject("Scripting.FileSystemObject").FileExists(candidate) Then
  shell.Run Chr(34) & candidate & Chr(34) & " " & gui, 0, False
  launched = True
End If

If Not launched Then
  candidate = appDir & "\runtime\pythonw.exe"
  If CreateObject("Scripting.FileSystemObject").FileExists(candidate) Then
    shell.Run Chr(34) & candidate & Chr(34) & " " & gui, 0, False
    launched = True
  End If
End If

If Not launched Then
  ' WindowStyle 0 = hidden; relies on pythonw in PATH.
  shell.Run "pythonw " & gui, 0, False
End If
