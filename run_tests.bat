@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m unittest discover -v -p "test_*.py"
pause
endlocal
