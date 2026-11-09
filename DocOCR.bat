@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -3 "%~dp0gui.py"
    goto :after_run
  )
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python -c "import tkinter" >nul 2>&1
  if %ERRORLEVEL%==0 (
    python "%~dp0gui.py"
    goto :after_run
  )
)

if exist "%~dp0runtime\pythonw.exe" (
  "%~dp0runtime\pythonw.exe" "%~dp0gui.py"
  goto :after_run
)

if exist "%~dp0runtime\python.exe" (
  "%~dp0runtime\python.exe" "%~dp0gui.py"
  goto :after_run
)

echo Не найден Python 3.9+ с tkinter.
echo Установите Python с https://www.python.org/downloads/
echo ^(галочка tcl/tk^) или используйте релиз с папкой runtime.
pause
exit /b 1

:after_run
if errorlevel 1 (
  echo.
  echo Программа завершилась с ошибкой.
  pause
)
endlocal
