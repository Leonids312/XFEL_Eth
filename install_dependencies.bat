@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m pip install --upgrade -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
echo Dependencies installed. Run run_xfel_app.bat to start.
goto end
:failed
echo Dependency installation failed.
:end
pause
endlocal
