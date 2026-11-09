@echo off
setlocal
cd /d "%~dp0"

rem Prefer pythonw so no console stays open behind the GUI.
rem "start" returns immediately and closes this bat window.

if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0gui.py"
  exit /b 0
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3w -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    start "" py -3w "%~dp0gui.py"
    exit /b 0
  )
)

where pythonw >nul 2>&1
if %ERRORLEVEL%==0 (
  pythonw -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    start "" pythonw "%~dp0gui.py"
    exit /b 0
  )
)

if exist "%~dp0runtime\pythonw.exe" (
  start "" "%~dp0runtime\pythonw.exe" "%~dp0gui.py"
  exit /b 0
)

rem Fallback: console Python (may flash a terminal).
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    start "" py -3 "%~dp0gui.py"
    exit /b 0
  )
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    start "" pythonw "%~dp0gui.py" 2>nul
    if errorlevel 1 start "" python "%~dp0gui.py"
    exit /b 0
  )
)

if exist "%~dp0runtime\python.exe" (
  start "" "%~dp0runtime\python.exe" "%~dp0gui.py"
  exit /b 0
)

echo Python 3.9+ with tkinter was not found.
echo Install Python from https://www.python.org/downloads/
echo ^(enable tcl/tk^) or use a release that includes the runtime folder.
pause
exit /b 1
