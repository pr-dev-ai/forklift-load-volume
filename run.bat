@echo off
rem Windows: creates a virtual environment, installs requirements, starts the app.
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

if exist ".venv\Scripts\python.exe" goto install
echo Creating virtual environment...
%PY% -m venv .venv
if errorlevel 1 goto fail

:install
echo Installing requirements...
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail

".venv\Scripts\python.exe" app.py %*
goto end

:fail
echo.
echo Setup failed. Please check that Python 3.9 or newer is installed and on PATH.
pause
exit /b 1

:end
endlocal
