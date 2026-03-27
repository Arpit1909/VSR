@echo off
setlocal EnableExtensions

REM Always run from this script's directory
cd /d "%~dp0"

set "VENV_DIR=videoEnv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "SETUP_FLAG=.venv_ready"
set "RUNTIME_FLAG=.venv_runtime"
set "PY_CMD="
set "LOCAL_CACHE_DIR=%CD%\.cache"
set "LOCAL_TEMP_DIR=%CD%\.temp"

if not exist "%LOCAL_CACHE_DIR%" mkdir "%LOCAL_CACHE_DIR%"
if not exist "%LOCAL_TEMP_DIR%" mkdir "%LOCAL_TEMP_DIR%"

REM Force Python/pip/torch temp and cache to current drive (D:)
set "TMP=%LOCAL_TEMP_DIR%"
set "TEMP=%LOCAL_TEMP_DIR%"
set "TMPDIR=%LOCAL_TEMP_DIR%"
set "PIP_CACHE_DIR=%LOCAL_CACHE_DIR%\pip"
set "TORCH_HOME=%LOCAL_CACHE_DIR%\torch"
set "HF_HOME=%LOCAL_CACHE_DIR%\huggingface"
set "XDG_CACHE_HOME=%LOCAL_CACHE_DIR%"

REM ---- Clean local temp ----
echo [INFO] Cleaning local temp on D: ...
if exist "%LOCAL_TEMP_DIR%" rmdir /s /q "%LOCAL_TEMP_DIR%" >nul 2>nul
mkdir "%LOCAL_TEMP_DIR%" >nul 2>nul
where forfiles >nul 2>nul
if not errorlevel 1 (
  if exist "%PIP_CACHE_DIR%" forfiles /p "%PIP_CACHE_DIR%" /s /d -14 /c "cmd /c del /f /q @path" >nul 2>nul
  if exist "%TORCH_HOME%" forfiles /p "%TORCH_HOME%" /s /d -30 /c "cmd /c del /f /q @path" >nul 2>nul
  if exist "%HF_HOME%" forfiles /p "%HF_HOME%" /s /d -30 /c "cmd /c del /f /q @path" >nul 2>nul
)

REM ---- Detect system specs ----
set "VSR_SYS_RAM_GB=0"
set "VSR_CPU_THREADS=0"
set "VSR_GPU_VRAM_GB=0"
set "RUNTIME_MODE=directml"
for /f %%A in ('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"') do set "VSR_SYS_RAM_GB=%%A"
for /f %%A in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty NumberOfLogicalProcessors)"') do set "VSR_CPU_THREADS=%%A"
for /f %%A in ('powershell -NoProfile -Command "$m=(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1); if($m){[int][math]::Round(([double]$m)/1024)} else {0}"') do set "VSR_GPU_VRAM_GB=%%A"
if not "%VSR_GPU_VRAM_GB%"=="0" set "RUNTIME_MODE=nvidia-cuda118"
echo [INFO] System detected: RAM=%VSR_SYS_RAM_GB%GB, CPU Threads=%VSR_CPU_THREADS%, GPU VRAM=%VSR_GPU_VRAM_GB%GB
REM If VRAM=0 (nvidia-smi missing) but you have ~8GB, uncomment so GUI picks Best Quality defaults:
REM set "VSR_GPU_VRAM_GB=8"

REM ---- Find Python ----
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys; print(sys.version)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.12"
)
if "%PY_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
  )
)
if "%PY_CMD%"=="" (
  echo [ERROR] Python 3.11+ not found. Python 3.12 is recommended.
  echo Install Python 3.12 and retry.
  pause
  exit /b 1
)

REM ---- Create venv if missing ----
if not exist "%VENV_PY%" (
  echo [INFO] Creating virtual environment...
  %PY_CMD% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

REM ---- Check venv Python version ----
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Existing venv uses Python older than 3.11. Recreating...
    rmdir /s /q "%VENV_DIR%"
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
      echo [ERROR] Failed to recreate virtual environment with Python 3.11+.
      pause
      exit /b 1
    )
    if exist "%SETUP_FLAG%" del /f /q "%SETUP_FLAG%" >nul 2>nul
    if exist "%RUNTIME_FLAG%" del /f /q "%RUNTIME_FLAG%" >nul 2>nul
  )
)

if not exist "%VENV_PY%" (
  echo [ERROR] Venv python not found: %VENV_PY%
  pause
  exit /b 1
)
if not exist "%VENV_PIP%" (
  echo [ERROR] Venv pip not found: %VENV_PIP%
  pause
  exit /b 1
)

REM ---- Handle flags ----
if /I "%~1"=="--setup"      goto SETUP
if /I "%~1"=="/setup"       goto SETUP
if /I "%~1"=="--reset"      goto RESET
if /I "%~1"=="/reset"       goto RESET
if /I "%~1"=="--clean-full" goto CLEAN_FULL
if /I "%~1"=="/clean-full"  goto CLEAN_FULL
if /I "%~1"=="--specs"      goto SPECS
if /I "%~1"=="/specs"       goto SPECS
if /I "%~1"=="--help"       goto HELP
if /I "%~1"=="/help"        goto HELP

if not exist "%SETUP_FLAG%" goto SETUP
goto RUN

:SETUP
echo [INFO] Installing base dependencies from requirements.txt...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.txt
  pause
  exit /b 1
)
call :INSTALL_RUNTIME
if errorlevel 1 (
  echo [ERROR] Runtime dependency installation failed.
  pause
  exit /b 1
)

REM ---- E2FGVI: install einops dependency ----
echo [INFO] Installing E2FGVI dependency: einops...
"%VENV_PY%" -m pip install einops
if errorlevel 1 (
  echo [WARN] Failed to install einops. E2FGVI mode may not work.
)

REM ---- E2FGVI: clone repo if missing ----
set "E2FGVI_REPO=backend\inpaint\e2fgvi_repo"
if not exist "%E2FGVI_REPO%\model\e2fgvi_hq.py" (
  where git >nul 2>nul
  if errorlevel 1 (
    echo [WARN] git not found. Cannot auto-clone E2FGVI repo.
    echo [WARN] Install git and run: git clone https://github.com/MCG-NKU/E2FGVI "%E2FGVI_REPO%"
  ) else (
    echo [INFO] Cloning E2FGVI repository...
    git clone https://github.com/MCG-NKU/E2FGVI "%E2FGVI_REPO%"
    if errorlevel 1 (
      echo [WARN] E2FGVI clone failed. Check internet connection and retry with: run_vsr.bat --setup
    ) else (
      echo [INFO] E2FGVI repo cloned successfully.
    )
  )
) else (
  echo [INFO] E2FGVI repo already present.
)

REM ---- E2FGVI: check model weights ----
set "E2FGVI_WEIGHTS=backend\models\e2fgvi\E2FGVI-HQ-CVPR22.pth"
if not exist "%E2FGVI_WEIGHTS%" (
  echo.
  echo [ACTION REQUIRED] E2FGVI model weights not found.
  echo [ACTION REQUIRED] Download from: https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3
  echo [ACTION REQUIRED] Save to: %CD%\%E2FGVI_WEIGHTS%
  echo [NOTE] mmcv is NOT required for this build (mmcv-free E2FGVI repo is used).
  echo.
) else (
  echo [INFO] E2FGVI model weights found.
)

echo ok> "%SETUP_FLAG%"

:RUN
if not exist "%RUNTIME_FLAG%" (
  echo [INFO] Runtime dependencies missing. Installing now...
  call :INSTALL_RUNTIME
  if errorlevel 1 (
    echo [ERROR] Runtime dependency installation failed.
    pause
    exit /b 1
  )
)
REM ---- E2FGVI: runtime checks ----
set "E2FGVI_REPO=backend\inpaint\e2fgvi_repo"
set "E2FGVI_WEIGHTS=backend\models\e2fgvi\E2FGVI-HQ-CVPR22.pth"
if not exist "%E2FGVI_REPO%\model\e2fgvi_hq.py" (
  echo [WARN] E2FGVI repo missing at %E2FGVI_REPO%
  echo [WARN] Run: run_vsr.bat --setup   to auto-clone it
)
if not exist "%E2FGVI_WEIGHTS%" (
  echo [WARN] E2FGVI weights missing: %E2FGVI_WEIGHTS%
  echo [WARN] Download from: https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3
)
if exist "backend\models\E2FGVI-HQ-CVPR22.pth" (
  if not exist "%E2FGVI_WEIGHTS%" (
    echo [WARN] Found weights at wrong location: backend\models\E2FGVI-HQ-CVPR22.pth
    echo [WARN] Move it to: backend\models\e2fgvi\E2FGVI-HQ-CVPR22.pth
  ) else (
    echo [INFO] Duplicate weights at backend\models\E2FGVI-HQ-CVPR22.pth - safe to delete ^(correct copy exists^)
  )
)

echo [INFO] Starting GUI...
echo [INFO] AUTO_SELECT_ALGORITHM=False ^(set True in config or GUI Settings for motion-based LaMa/E2FGVI/ProPainter^)
echo [INFO] E2FGVI-HQ ^(8GB^): e2fgvi_hq.py + E2FGVI-HQ-CVPR22.pth  NEIGHBOR=35  REF_STRIDE=2  MAX_LOAD=60  STREAM=32  MAX_CROP=1600  FP16=True
echo [INFO] Encode: OUTPUT_CRF=14  OUTPUT_PRESET=slow  ^(USE_H264=True^)
echo [INFO] Quality defaults ^(backend\config.py^): ANTIFLICKER=0.97  FEATHER=3  SUBTITLE_PAD=20px  LaMa final check OFF
echo [INFO] GUI Settings ^(Quality frame, 2nd row^): E2FGVI/feather/pad/antiflicker/manual-OCR — saves gui_user_settings.json ^(overrides config on each Run^)
echo [INFO] MANUAL_BOXES_ONLY default False; E2FGVI_FORCE_SCENE_DETECT default False ^(scene scan can sit at 0%% progress^)
echo [INFO] Detection: CLAHE pre-processing for night/dark scenes before multi-pass detection
echo [INFO] Detection: multi-pass ^(CLAHE+original / CLAHE+inverted^), DB thresh=0.2/0.45, unclip=2.0
echo [INFO] Detection: covers any-color text + colored background boxes on day and night footage
echo [INFO] E2FGVI-HQ: one union-mask pass per subtitle segment + batched crossfade
echo [INFO] Mask: horizontal morphological closing fills gaps between characters in color bg boxes
echo [INFO] Shortcuts: Space=Play/Pause, Left/Right=step, Shift+Left/Right=jump 10, Ctrl+Z=Undo Box
echo [INFO] Draw boxes directly over subtitles ^(no auto-expand; use sliders to refine^)
echo [INFO] Multi-box: draw 2-3+ boxes for top/bottom/watermark removal simultaneously
echo [INFO] GUI defaults ^(with detected RAM=%VSR_SYS_RAM_GB%GB VRAM=%VSR_GPU_VRAM_GB%GB CPU=%VSR_CPU_THREADS%^): Two-Pass Best + Best Quality when VRAM^>=8
echo [INFO] Resume Last + subtitle timeline + Preview/Final/Two-Pass pass mode available in GUI.
echo [INFO] Quality presets: Recommended / BQ Optimized / Max Quality ^(overwrite numbers until you Apply Settings again^)
echo [INFO] Live Elapsed/ETA line uses real measured progress ^(more stable than tqdm ETA^).
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512"
REM Unbuffered stdout/stderr so tracebacks show immediately if the GUI errors
set "PYTHONUNBUFFERED=1"
"%VENV_PY%" gui.py
if errorlevel 1 (
  echo.
  echo [ERROR] GUI exited with an error.
  echo.
  echo [TIP 1]  Black screen after Open?   Fixed: PNG encoding used for preview ^(JPEG caused silent TclError^)
  echo [TIP 2]  E2FGVI repo missing?       Run: run_vsr.bat --setup  ^(auto-clones repo + installs einops^)
  echo [TIP 3]  E2FGVI weights missing?    Download E2FGVI-HQ-CVPR22.pth to backend\models\e2fgvi\
  echo [TIP 4]  GPU out of memory / stuck?  Lower E2FGVI max batch in GUI Settings or backend\config.py — check VRAM with nvidia-smi
  echo [TIP 5]  Subtitles not removed?     Make sure each box center covers text - use multi-box for top+bottom subs
  echo [TIP 6]  Night scene text missed?   CLAHE contrast boost applied before detection automatically
  echo [TIP 7]  Colored text missed?       Multi-pass ^(CLAHE+original / CLAHE+inverted^) catches all colors
  echo [TIP 8]  Still flickering?          Lower antiflicker alpha in GUI Settings or backend\config.py ^(e.g. 0.85^)
  echo [TIP 9]  Output blurry?             Tighten boxes; lower pad/feather in Settings; reduce MAX_CROP_SIDE on 8GB
  echo [TIP 10] Want auto algo by motion?  Set AUTO_SELECT_ALGORITHM=True in backend\config.py
  echo [TIP 11] Broken deps?               Run: run_vsr.bat --setup
  echo [TIP 12] Reset flags only?          Run: run_vsr.bat --reset
  echo [TIP 13] Check hardware specs?      Run: run_vsr.bat --specs
  pause
)
goto END

:RESET
echo [INFO] Resetting setup flags (venv kept, deps will reinstall on next run)...
if exist "%SETUP_FLAG%"   del /f /q "%SETUP_FLAG%"   >nul 2>nul
if exist "%RUNTIME_FLAG%" del /f /q "%RUNTIME_FLAG%" >nul 2>nul
echo [INFO] Done. Run run_vsr.bat to reinstall dependencies.
goto END

:CLEAN_FULL
echo [INFO] Full cleanup started...
if exist "%LOCAL_TEMP_DIR%" rmdir /s /q "%LOCAL_TEMP_DIR%" >nul 2>nul
if exist "%LOCAL_CACHE_DIR%" rmdir /s /q "%LOCAL_CACHE_DIR%" >nul 2>nul
mkdir "%LOCAL_TEMP_DIR%" >nul 2>nul
mkdir "%LOCAL_CACHE_DIR%" >nul 2>nul
if exist "%SETUP_FLAG%"   del /f /q "%SETUP_FLAG%"   >nul 2>nul
if exist "%RUNTIME_FLAG%" del /f /q "%RUNTIME_FLAG%" >nul 2>nul
echo [INFO] Full cleanup complete.
goto END

:SPECS
echo [INFO] Hardware profile summary:
echo [INFO]   RAM:          %VSR_SYS_RAM_GB% GB
echo [INFO]   CPU Threads:  %VSR_CPU_THREADS%
echo [INFO]   GPU VRAM:     %VSR_GPU_VRAM_GB% GB
echo [INFO]   Runtime mode: %RUNTIME_MODE%
for /f "tokens=*" %%A in ('powershell -NoProfile -Command "nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1"') do echo [INFO]   GPU Name:     %%A
goto END

:HELP
echo.
echo Usage:
echo   run_vsr.bat              ^(normal run^)
echo   run_vsr.bat --setup      ^(reinstall all deps + clone E2FGVI repo + check weights^)
echo   run_vsr.bat --reset      ^(clear flags, reinstalls deps on next run^)
echo   run_vsr.bat --clean-full ^(delete .temp and .cache^)
echo   run_vsr.bat --specs      ^(print detected hardware and exit^)
echo.
echo E2FGVI setup ^(required one-time^):
echo   1. Run:  run_vsr.bat --setup            ^(auto-clones repo, installs einops^)
echo   2. Download weights ^(~200MB^):
echo      https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3
echo   3. Save to:  backend\models\e2fgvi\E2FGVI-HQ-CVPR22.pth
echo   NOTE: mmcv is NOT required for this build.
echo.
echo Default pipeline ^(backend\config.py^):
echo   AUTO_SELECT_ALGORITHM = False — set True for LaMa / E2FGVI / ProPainter by measured motion
echo   MODE = E2FGVI ^(HQ weights only^)
echo   Optional override: gui_user_settings.json ^(written by GUI Settings; merged each Run^)
echo.
echo E2FGVI-HQ defaults ^(8GB VRAM class, backend\config.py^):
echo   E2FGVI_USE_HQ          = True ^(only HQ supported in this build^)
echo   E2FGVI_NEIGHBOR_LENGTH = 35
echo   E2FGVI_REF_STRIDE      = 2
echo   E2FGVI_MAX_LOAD_NUM    = 60
echo   E2FGVI_STREAM_MAX_LOAD = 32
echo   E2FGVI_MAX_CROP_SIDE   = 1600
echo   E2FGVI_CROP_MARGIN     = 128
echo   E2FGVI_USE_FP16        = True
echo.
echo Quality defaults ^(backend\config.py^):
echo   SUBTITLE_AREA_DEVIATION_PIXEL = 20
echo   MASK_FEATHER_RADIUS           = 3
echo   ANTIFLICKER_ALPHA             = 0.97
echo   OUTPUT_CRF                    = 14
echo   OUTPUT_PRESET                 = slow
echo   ENABLE_FAST_FINAL_CHECK       = False ^(use GUI Strict only if text remains^)
echo.
echo GUI tips:
echo   - Settings ^(Quality frame^): tune E2FGVI, feather, pad, antiflicker, manual-OCR flags; saves gui_user_settings.json
echo   - Default Final Check: Off — avoids LaMa-on-E2FGVI flicker
echo   - Default pass: Two-Pass Best; default quality: Best Quality when VRAM^>=8GB
echo   - Draw 2-3+ boxes for simultaneous top subtitle + bottom subtitle + watermark removal
echo   - Any-color text: CLAHE + multi-pass detection catches subs on day and night footage
echo   - Colored background box: horizontal mask closing fills gaps between characters automatically
echo   - Use Resume Last to continue from checkpoint after crash/stop
echo   - Timeline bar shows detected subtitle intervals
echo   - Live ETA: shown in GUI as Elapsed/ETA/Stage ^(based on real elapsed time and progress^)
echo.
goto END

:INSTALL_RUNTIME
set "RUNTIME_MODE=directml"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "RUNTIME_MODE=nvidia-cuda118"
echo [INFO] Detected runtime: %RUNTIME_MODE%
if /I "%RUNTIME_MODE%"=="nvidia-cuda118" (
  echo [INFO] Installing NVIDIA CUDA 11.8 runtime packages...
  "%VENV_PY%" -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  if errorlevel 1 exit /b 1
  "%VENV_PY%" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
  if errorlevel 1 exit /b 1
) else (
  echo [INFO] Installing DirectML runtime packages...
  if exist "requirements_directml.txt" (
    "%VENV_PY%" -m pip install -r requirements_directml.txt
    if errorlevel 1 exit /b 1
  ) else (
    "%VENV_PY%" -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    if errorlevel 1 exit /b 1
    "%VENV_PY%" -m pip install torch_directml==0.2.5.dev240914
    if errorlevel 1 exit /b 1
  )
)
REM ---- E2FGVI extra dep ----
echo [INFO] Installing E2FGVI dependency: einops...
"%VENV_PY%" -m pip install einops
echo %RUNTIME_MODE%> "%RUNTIME_FLAG%"
exit /b 0

:END
endlocal
