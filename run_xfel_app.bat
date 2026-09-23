@echo off
setlocal
cd /d "%~dp0"

py -3.11 -c "import h5py, cv2, matplotlib, openpyxl, PIL, numpy, tksheet" >nul 2>nul
if not errorlevel 1 goto run_app

echo Installing missing XFEL application dependencies...
py -3.11 -m pip install -r "%~dp0requirements.txt"
if not errorlevel 1 goto run_app

echo.
echo Dependency installation failed.
echo Run install_dependencies.bat, then try again.
pause
exit /b 1

:run_app
py -3.11 "%~dp0xfel_damage_threshold_app.py" %* >"%~dp0xfel_startup_error.log" 2>&1
set "APP_EXIT_CODE=%errorlevel%"
if "%APP_EXIT_CODE%"=="0" goto end

echo.
echo XFEL application exited with code %APP_EXIT_CODE%.
echo Captured error:
type "%~dp0xfel_startup_error.log"
pause

:end
endlocal & exit /b %APP_EXIT_CODE%
