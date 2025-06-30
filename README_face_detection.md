# Face Detection System

A comprehensive real-time face detection system using OpenCV's Haar Cascade classifiers.

## Features

- **Real-time Processing**: Works with both video files and live webcam feeds
- **Multiple Detection Models**: Uses both frontal and profile face detection
- **Eye Detection**: Detects eyes within detected faces for better accuracy
- **Interactive Controls**: Pause, resume, save frames, and adjust sensitivity on-the-fly
- **Statistics Display**: Shows real-time FPS, detection counts, and performance metrics
- **Overlap Removal**: Intelligent algorithm to remove duplicate detections
- **Professional UI**: Clean bounding boxes with confidence indicators

## Quick Start

### 1. Setup Environment
Install the required dependencies:
```bash
# Install OpenCV with extra modules
pip install opencv-contrib-python>=4.8.0

# Install NumPy
pip install numpy>=1.24.0

# Install additional utilities (optional)
pip install imutils>=0.5.4
```

### 2. Run Face Detection

**For webcam:**
```bash
python face_detection_main.py --camera 0
```

**For video file (using included sample):**
```bash
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"
```

**Save output:**
```bash
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save
```

## Using Included Sample Video

The workspace includes a high-quality sample video file perfect for testing face detection:

- **File**: `input/1338598-hd_1920_1080_30fps.mp4`
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Purpose**: Ideal for testing face detection algorithms

### Quick Test Commands:
```bash
# Test face detection on sample video
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Test and save output
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save

# Test with high sensitivity
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --sensitivity high --save
```

## Test Results

### Verified System Performance

**Test Environment:**
- OS: Windows 11
- Camera: Integrated webcam (1920x1080)
- Sample Video: Full HD (1920x1080 @ 30 FPS, 223 frames)

**Initialization Output:**
```
🎯 Face Detection System Starting...
📋 Configuration:
   Sensitivity: medium
   Scale factor: 1.1
   Min neighbors: 5
   Min size: 30x30
🔥 Initializing Face Detection System...
✅ Loaded cascade: haarcascade_frontalface_default.xml
✅ Loaded cascade: haarcascade_profileface.xml
✅ Loaded cascade: haarcascade_eye.xml
✅ Face Detection System initialized successfully!
```

**Video Processing:**
```
📹 Video Info: 1920x1080 @ 30.0 FPS
📊 Total frames: 223
💾 Saving output to: output/detected_faces.mp4
🎬 Starting face detection... Press 'q' to quit
```

**Camera Test Results:**
```
🔍 Testing camera 0 resolutions...
📊 Supported resolutions for camera 0:
   1920x1080
💡 Recommended: --width 1920 --height 1080
```

**Performance Metrics:**
- ✅ Real-time processing on Full HD video
- ✅ Multiple face detection (frontal + profile)
- ✅ Eye detection within faces
- ✅ Output video generation successful
- ✅ Interactive controls responsive
- ✅ Statistics tracking accurate

## Command Line Options

```
usage: face_detection_main.py [-h] (--input INPUT | --camera CAMERA) [--save]
                              [--output OUTPUT] [--sensitivity {low,medium,high}]
                              [--min-size WIDTH HEIGHT] [--width WIDTH] [--height HEIGHT]
                              [--scale SCALE] [--no-flip] [--test-camera ID]

Real-Time Face Detection System

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input video file path
  --camera CAMERA       Camera index (0 for default camera)
  --save                Save output video
  --output OUTPUT       Output video path (default: output/detected_faces.mp4)
  --sensitivity {low,medium,high}
                        Detection sensitivity (default: medium)
  --min-size WIDTH HEIGHT
                        Minimum face size (default: 30 30)
  --width WIDTH         Camera width resolution (default: 1280)
  --height HEIGHT       Camera height resolution (default: 720)
  --scale SCALE         Display scale factor (0.7 = 70% size)
  --no-flip             Disable camera horizontal flip
  --test-camera ID      Test camera resolutions and exit
```

## Interactive Controls

While the detection is running, you can use these keyboard shortcuts:

- **'q'** - Quit the application
- **'p'** - Pause/Resume processing
- **'s'** - Save current frame as image
- **'r'** - Reset statistics counters
- **'+'** - Increase detection sensitivity
- **'-'** - Decrease detection sensitivity

## Sensitivity Levels

- **Low**: More conservative detection (fewer false positives, may miss some faces)
- **Medium**: Balanced detection (recommended for most cases)
- **High**: More aggressive detection (catches more faces, may have false positives)

## Detection Types

The system detects:
- **Frontal Faces** (shown in green rectangles)
- **Profile Faces** (shown in blue rectangles)
- **Eyes** (shown in yellow rectangles within faces)

## Output Files

When saving is enabled, the system creates:
- **Video Output**: `output/detected_faces.mp4`
- **Frame Snapshots**: `output/face_frame_XXXXXX.jpg` (when pressing 's')

## Performance Tips

1. **For better performance:**
   - Use lower resolution videos or webcam settings
   - Increase minimum face size for distant objects
   - Use medium or low sensitivity
   - Process the included sample video first to test performance

2. **For better accuracy:**
   - Ensure good lighting conditions
   - Use high sensitivity setting
   - Process at original resolution
   - Test with the included HD sample video

3. **For real-time applications:**
   - Use webcam input with appropriate resolution
   - Test camera capabilities with `--test-camera 0`
   - Optimize detection parameters based on your use case

## Technical Details

### Detection Algorithm
- Uses OpenCV's Haar Cascade Classifiers
- Multiple cascade models for comprehensive detection
- Histogram equalization for improved contrast
- Non-maximum suppression for overlap removal

### Performance Metrics
- Real-time FPS calculation
- Detection statistics and counters
- Processing time per frame
- Average faces per frame

### File Structure
```
Vision/
├── face_detection_main.py          # Main face detection script
├── person_detection_main.py        # Person detection script (YOLO-based)
├── requirements.txt                # Python dependencies
├── yolov8n.pt                      # YOLO model file
├── input/                          # Input video files
│   ├── 1338598-hd_1920_1080_30fps.mp4  # Sample HD video for testing
│   └── README.txt                  # Input folder documentation
└── output/                         # Output files
    ├── detected_faces.mp4          # Face detection output
    ├── detected_people.mp4         # Person detection output
    ├── face_frame_XXXXXX.jpg       # Saved face detection frames
    └── README.txt                  # Output folder documentation
```

## Requirements

- Python 3.8+
- OpenCV 4.8.0+ (with contrib modules)
- NumPy 1.24.0+
- Cross-platform (Windows, macOS, Linux)

## Troubleshooting

### Common Issues

1. **"Could not load cascade" error:**
   - Ensure OpenCV is properly installed
   - Try reinstalling with: `pip install opencv-contrib-python`

2. **Camera not found:**
   - Check camera index (try 0, 1, 2...)
   - Ensure camera is not used by another application

3. **Poor detection quality:**
   - Adjust sensitivity settings
   - Ensure good lighting conditions
   - Try different minimum face sizes

4. **Low performance:**
   - Reduce video resolution
   - Increase minimum face size
   - Use lower sensitivity

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Test with different input sources
4. Adjust detection parameters

## Examples

### Basic Usage
```bash
# Webcam detection
python face_detection_main.py --camera 0

# Process the included sample video
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Process sample video and save output
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save

# Process any video file with high sensitivity
python face_detection_main.py --input "my_video.mp4" --save --sensitivity high
```

### Advanced Usage
```bash
# Test camera capabilities first
python face_detection_main.py --test-camera 0

# HD webcam with custom settings
python face_detection_main.py --camera 0 --width 1920 --height 1080 --no-flip

# Webcam with custom minimum face size
python face_detection_main.py --camera 0 --min-size 50 50

# Process video with custom output path
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save --output "my_face_output.mp4"

# Scaled display for smaller window
python face_detection_main.py --camera 0 --scale 0.5
```

## License

This project uses OpenCV which is licensed under the Apache 2.0 License.
