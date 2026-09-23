@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m unittest -v test_xfel_core.py
pause
endlocal
