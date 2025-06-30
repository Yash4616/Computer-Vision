@echo off
REM Vision Detection Systems - Setup Instructions
REM ==============================================
REM This file provides setup instructions. For actual setup, use pip commands.

echo ================================================
echo    Vision Detection Systems Setup Guide
echo ================================================
echo.
echo This workspace contains two detection systems:
echo   1. Person Detection (YOLO-based)
echo   2. Face Detection (OpenCV-based)
echo.
echo SETUP INSTRUCTIONS:
echo ==================
echo.
echo 1. Install Python dependencies:
echo    pip install -r requirements.txt
echo.
echo    OR install individually:
echo    pip install opencv-contrib-python^>=4.8.0
echo    pip install ultralytics^>=8.0.0
echo    pip install torch^>=2.0.0
echo    pip install numpy^>=1.24.0
echo.
echo 2. Test installations:
echo    python -c "import cv2; print('OpenCV OK')"
echo    python -c "import ultralytics; print('YOLO OK')"
echo.
echo 3. Run detection systems:
echo.
echo    Person Detection:
echo    python person_detection_main.py --camera 0
echo    python person_detection_main.py --input "input/Video.mp4"
echo.
echo    Face Detection:
echo    python face_detection_main.py --camera 0
echo    python face_detection_main.py --input "input/Video.mp4"
echo.
echo ================================================
echo For detailed documentation, see:
echo   - README.md (main overview)
echo   - README_person_detection.md (person detection guide)
echo   - README_face_detection.md (face detection guide)
echo ================================================
echo.
pause
