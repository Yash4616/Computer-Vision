@echo off
REM Vision Detection Systems - Quick Launcher
REM ==========================================
REM This file provides quick launch options. For full control, use Python directly.

echo ================================================
echo      Vision Detection Systems Launcher
echo ================================================
echo.
echo Select a detection system to run:
echo.
echo 1. Person Detection - Webcam
echo 2. Person Detection - Sample Video
echo 3. Face Detection - Webcam  
echo 4. Face Detection - Sample Video
echo 5. Test Camera Capabilities
echo 6. Show Installation Status
echo 7. Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Starting Person Detection with webcam...
    python person_detection_main.py --camera 0
) else if "%choice%"=="2" (
    echo.
    echo Starting Person Detection with sample video...
    python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"
) else if "%choice%"=="3" (
    echo.
    echo Starting Face Detection with webcam...
    python face_detection_main.py --camera 0
) else if "%choice%"=="4" (
    echo.
    echo Starting Face Detection with sample video...
    python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"
) else if "%choice%"=="5" (
    echo.
    echo Testing camera capabilities...
    python person_detection_main.py --test-camera 0
) else if "%choice%"=="6" (
    echo.
    echo Checking installation status...
    echo.
    echo Dependencies Status:
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>nul || echo "OpenCV: Not installed"
    python -c "import ultralytics; print('YOLO: Available')" 2>nul || echo "YOLO: Not installed"
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo "PyTorch: Not installed"
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul || echo "NumPy: Not installed"
    echo.
    echo Hardware Status:
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>nul || echo "CUDA: Not available"
    echo.
) else if "%choice%"=="7" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
)

echo.
echo ================================================
echo Detection completed!
echo For more options, run Python scripts directly:
echo   python person_detection_main.py --help
echo   python face_detection_main.py --help
echo ================================================
pause
