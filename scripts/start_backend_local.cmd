@echo off
setlocal
for %%I in ("%~dp0..\..\.venv\Scripts") do set "VENV_BIN=%%~fI"
for %%I in ("%~dp0..\backend.local.stdout.log") do set "STDOUT_LOG=%%~fI"
for %%I in ("%~dp0..\backend.local.stderr.log") do set "STDERR_LOG=%%~fI"
set "PATH=%VENV_BIN%;%PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_backend.ps1" %* 1>>"%STDOUT_LOG%" 2>>"%STDERR_LOG%"
