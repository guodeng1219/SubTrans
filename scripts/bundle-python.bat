@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

REM ============================================================
REM Build python-bundle for distributing with the app.
REM Downloads Python embeddable + pip installs all GPU packages.
REM Run: scripts\bundle-python.bat    (or just double-click this file)
REM Takes ~10-30 minutes first time (torch is 2.5GB).
REM ============================================================

set "BUNDLE=src-tauri\python-bundle"
set "PY_EXE=%BUNDLE%\python.exe"
set "PYTHON_ZIP=python-3.12.9-embed-amd64.zip"
set "PYTHON_VERSION=3.12.9"
set "PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/"
set "PIP_HOST=mirrors.aliyun.com"

REM ---- already done? ----
if exist "%PY_EXE%" (
    if exist "%BUNDLE%\Lib\site-packages\torch\__init__.py" (
        echo [OK] python-bundle already complete.
        exit /b 0
    )
)

echo.
echo ========================================
echo   Building python-bundle
echo   This only needs to be done ONCE.
echo   First run takes 10-30 minutes.
echo ========================================
echo.

if not exist "%BUNDLE%" mkdir "%BUNDLE%"

REM ============================================================
REM 1. Download Python embeddable
REM ============================================================
set "ZIP=%BUNDLE%\%PYTHON_ZIP%"
call :download "https://mirrors.tuna.tsinghua.edu.cn/python/%PYTHON_VERSION%/%PYTHON_ZIP%" "%ZIP%" "Python embeddable (Tsinghua)"
if not exist "%ZIP%" (
    call :download "https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_ZIP%" "%ZIP%" "Python embeddable (python.org)"
)
if not exist "%ZIP%" (
    echo [ERROR] Could not download Python. Check your internet connection.
    pause
    exit /b 1
)

REM ============================================================
REM 2. Extract
REM ============================================================
echo Extracting Python...
%SystemRoot%\System32\tar.exe -xf "%ZIP%" -C "%BUNDLE%" 2>nul
if errorlevel 1 (
    echo tar not available, trying PowerShell...
    powershell -NoProfile -Command "Expand-Archive -Path '%ZIP%' -DestinationPath '%BUNDLE%' -Force" 2>nul
    if errorlevel 1 (
        echo [ERROR] Cannot extract zip. Install 7-Zip or enable tar.
        pause
        exit /b 1
    )
)
del "%ZIP%" 2>nul

REM ============================================================
REM 3. Enable "import site" in ._pth (required for pip)
REM ============================================================
for %%f in ("%BUNDLE%\python3*._pth") do set "PTH=%%f"
if exist "!PTH!" (
    echo Configuring pip...
    findstr /v /c:"#import site" "!PTH!" > "%BUNDLE%\_pth_tmp"
    echo import site>> "%BUNDLE%\_pth_tmp"
    move /y "%BUNDLE%\_pth_tmp" "!PTH!" >nul
)

REM ============================================================
REM 4. Install pip
REM ============================================================
echo Installing pip...
call :download "https://mirrors.aliyun.com/pypi/get-pip.py" "%BUNDLE%\get-pip.py" "get-pip.py"
if not exist "%BUNDLE%\get-pip.py" (
    call :download "https://bootstrap.pypa.io/get-pip.py" "%BUNDLE%\get-pip.py" "get-pip.py (bootstrap)"
)
if exist "%BUNDLE%\get-pip.py" (
    "%PY_EXE%" "%BUNDLE%\get-pip.py"
    if errorlevel 1 echo [WARN] get-pip.py had errors, continuing...
    del "%BUNDLE%\get-pip.py" 2>nul
)

REM ============================================================
REM 5. Preflight: certifi
REM ============================================================
echo pip install certifi ^(network check^)...
"%PY_EXE%" -m pip install -i %PIP_MIRROR% --trusted-host %PIP_HOST% --no-cache-dir certifi
if errorlevel 1 (
    echo [ERROR] Cannot reach Aliyun PyPI mirror. Check your network.
    pause
    exit /b 1
)

REM ============================================================
REM 6. CPU torch (~200MB) - CUDA version downloaded at runtime if GPU detected
REM ============================================================
echo.
echo ========================================
echo   Installing CPU torch (~200MB)
echo   (CUDA torch will be downloaded at runtime if GPU detected)
echo ========================================
echo.

"%PY_EXE%" -m pip install -i %PIP_MIRROR% torch torchaudio
if errorlevel 1 (
    echo [ERROR] CPU torch install failed.
    pause
    exit /b 1
)

REM ============================================================
REM 7. Remaining packages (faster-whisper + demucs + audio-separator + soundfile)
REM ============================================================
echo.
echo Installing faster-whisper / demucs / audio-separator / soundfile...
"%PY_EXE%" -m pip install -i %PIP_MIRROR% faster-whisper demucs audio-separator soundfile
if errorlevel 1 (
    echo [ERROR] Package install failed.
    pause
    exit /b 1
)

REM ============================================================
REM 8. Cleanup: remove pip cache and __pycache__ to reduce bundle size
REM ============================================================
echo Cleaning up pip cache and __pycache__...
"%PY_EXE%" -m pip cache purge 2>nul
for /d /r "%BUNDLE%\Lib" %%d in (__pycache__) do rd /s /q "%%d" 2>nul
echo Cleanup done.

echo.
echo ========================================
echo   python-bundle ready!
echo   Size: ~500MB (will compress to ~250MB in NSIS)
echo ========================================
exit /b 0

REM ============================================================
REM Helper: download URL to file
REM ============================================================
:download
echo Downloading %~3...
echo   %~1
REM try curl first
%SystemRoot%\System32\curl.exe -L -o "%~2" "%~1" --connect-timeout 30 --max-time 300 -s -S 2>nul
if exist "%~2" exit /b 0
REM fallback: certutil (Windows built-in)
certutil -urlcache -split -f "%~1" "%~2" >nul 2>nul
if exist "%~2" exit /b 0
echo   [FAIL] Could not download: %~1
exit /b 1
