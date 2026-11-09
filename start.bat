@echo off
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0gui.py"
  exit /b 0
)

if exist "%~dp0DocOCR.vbs" (
  wscript "%~dp0DocOCR.vbs"
  exit /b 0
)

if exist "%~dp0runtime\pythonw.exe" (
  start "" "%~dp0runtime\pythonw.exe" "%~dp0gui.py"
  exit /b 0
)

where pythonw >nul 2>&1
if %ERRORLEVEL%==0 (
  start "" pythonw "%~dp0gui.py"
  exit /b 0
)

echo Не найден pythonw. Сначала установите библиотеки или Python 3.9+.
pause
exit /b 1
