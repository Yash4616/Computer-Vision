# Person Detection System

A comprehensive real-time person detection system using YOLOv8 (You Only Look Once) neural network.

## Features

- **Real-time Processing**: Works with both video files and live webcam feeds
- **YOLO-based Detection**: Uses state-of-the-art YOLOv8 model for accurate person detection
- **GPU/CPU Support**: Automatic hardware detection and optimization
- **Interactive Controls**: Pause, resume, save frames, and adjust parameters on-the-fly
- **Statistics Display**: Shows real-time FPS, detection counts, and performance metrics
- **Professional UI**: Clean bounding boxes with confidence scores
- **Advanced Camera Controls**: Resolution testing, auto-configuration, and fallback options

## Quick Start

### 1. Setup Environment
Install the required dependencies:
```bash
# Install core requirements
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install numpy>=1.24.0

# Install additional utilities
pip install filterpy==1.4.5
pip install scikit-image==0.21.0
pip install imutils==0.5.4
```

### 2. Run Person Detection

**For webcam:**
```bash
python person_detection_main.py --camera 0
```

**For video file (using included sample):**
```bash
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"
```

**Save output:**
```bash
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/my_detection.mp4"
```

## Using Included Sample Video

The workspace includes a high-quality sample video file perfect for testing person detection:

- **File**: `input/1338598-hd_1920_1080_30fps.mp4`
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Purpose**: Ideal for testing person detection algorithms

### Quick Test Commands:
```bash
# Test person detection on sample video
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Test and save output
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/detected_people.mp4"

# Test with higher confidence threshold
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --conf 0.5 --output "output/high_conf_detection.mp4"
```

## Test Results

### Verified System Performance

**Test Environment:**
- OS: Windows 11  
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6.0GB)
- Camera: Integrated webcam (1920x1080)
- Sample Video: Full HD (1920x1080 @ 30 FPS, 223 frames)

**GPU Detection & Initialization:**
```
🔥 Initializing Person Detection System...
🚀 GPU detected: NVIDIA GeForce RTX 4050 Laptop GPU (6.0GB)
🖥️  Using device: cuda:0
📥 Loading YOLO model: yolov8n.pt
✅ Model loaded successfully on cuda:0
✅ Person Detection System Ready!
```

**Camera Test Results:**
```
🔍 Testing camera 0 resolutions...
📊 Supported resolutions for camera 0:
   1920x1080
💡 Recommended: --width 1920 --height 1080
```

**Performance Metrics:**
- ✅ GPU acceleration with CUDA support
- ✅ YOLOv8 nano model loaded successfully
- ✅ Real-time processing capability
- ✅ High-resolution camera support (1920x1080)
- ✅ Automatic hardware optimization
- ✅ Output video generation successful

**Hardware Optimization:**
- **GPU Memory**: 6.0GB VRAM detected and utilized
- **Processing**: CUDA-accelerated inference
- **Model**: YOLOv8n (6.2MB) optimized for speed
- **Resolution**: Full HD support confirmed

## Command Line Options

```
usage: person_detection_main.py [-h] [--input INPUT | --camera CAMERA] [--conf CONF]
                               [--iou IOU] [--model MODEL] [--width WIDTH] [--height HEIGHT]
                               [--scale SCALE] [--no-flip] [--output OUTPUT] [--no-display]
                               [--test-camera ID]

Real-Time Person Detection with YOLO

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to input video file
  --camera CAMERA       Camera device ID (0, 1, etc.)
  --conf CONF           Confidence threshold (0.0-1.0, default: 0.3)
  --iou IOU             IoU threshold for NMS (default: 0.45)
  --model MODEL         YOLO model path (default: yolov8n.pt)
  --width WIDTH         Camera width resolution (default: 1920)
  --height HEIGHT       Camera height resolution (default: 1080)
  --scale SCALE         Display scale factor (default: 0.7)
  --no-flip             Disable camera horizontal flip
  --output OUTPUT       Output video path (optional)
  --no-display         Disable video display
  --test-camera ID      Test camera resolutions and exit
```

## Interactive Controls

While the detection is running, you can use these keyboard shortcuts:

- **'q'** - Quit the application
- **'p'** - Pause/Resume processing
- **'s'** - Save current frame as image
- **'r'** - Reset statistics counters
- **'+'** - Increase confidence threshold
- **'-'** - Decrease confidence threshold
- **'c'** - Toggle confidence display
- **'f'** - Toggle FPS display

## Detection Parameters

### Confidence Threshold
- **Low (0.1-0.3)**: More detections, may include false positives
- **Medium (0.3-0.5)**: Balanced detection (recommended for most cases)
- **High (0.5-0.8)**: Fewer detections, higher accuracy

### IoU Threshold
- Controls overlap detection for Non-Maximum Suppression
- Default: 0.45 (good balance between removing duplicates and keeping valid detections)

## YOLO Models

The system supports different YOLO model variants:

- **YOLOv8n** (`yolov8n.pt`) - Nano: Fastest, smallest model
- **YOLOv8s** (`yolov8s.pt`) - Small: Good balance of speed and accuracy
- **YOLOv8m** (`yolov8m.pt`) - Medium: Better accuracy, slower
- **YOLOv8l** (`yolov8l.pt`) - Large: High accuracy, requires more resources
- **YOLOv8x** (`yolov8x.pt`) - Extra Large: Best accuracy, slowest

## Output Files

When saving is enabled, the system creates:
- **Video Output**: Specified path or `output/detected_people.mp4`
- **Frame Snapshots**: `output/person_frame_XXXXXX.jpg` (when pressing 's')

## Performance Tips

1. **For better performance:**
   - Use YOLOv8n (nano) model for real-time applications
   - Reduce camera resolution for webcam input
   - Use GPU if available (automatic detection)
   - Process the included sample video first to test performance

2. **For better accuracy:**
   - Use larger YOLO models (YOLOv8m, YOLOv8l, YOLOv8x)
   - Increase confidence threshold to reduce false positives
   - Ensure good lighting conditions
   - Test with the included HD sample video

3. **For real-time applications:**
   - Use webcam input with appropriate resolution
   - Test camera capabilities with `--test-camera 0`
   - Optimize model and parameters based on your hardware

## Technical Details

### Detection Algorithm
- Uses YOLOv8 (You Only Look Once) neural network
- Single-pass detection for real-time performance
- Non-Maximum Suppression for duplicate removal
- Automatic GPU/CPU detection and optimization

### Performance Metrics
- Real-time FPS calculation
- Detection statistics and counters
- Processing time per frame
- GPU/CPU utilization information

### Hardware Requirements
- **Minimum**: CPU-only, 4GB RAM
- **Recommended**: NVIDIA GPU with CUDA support, 8GB RAM
- **Optimal**: RTX 30/40 series GPU, 16GB RAM

### File Structure
```
Vision/
├── person_detection_main.py        # Main person detection script
├── face_detection_main.py          # Face detection script (OpenCV-based)
├── requirements.txt                # Python dependencies
├── yolov8n.pt                      # YOLO nano model file
├── input/                          # Input video files
│   ├── 1338598-hd_1920_1080_30fps.mp4  # Sample HD video for testing
│   └── README.txt                  # Input folder documentation
└── output/                         # Output files
    ├── detected_people.mp4         # Person detection output
    ├── detected_faces.mp4          # Face detection output
    ├── person_frame_XXXXXX.jpg     # Saved person detection frames
    └── README.txt                  # Output folder documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV 4.8.0+
- CUDA (optional, for GPU acceleration)
- Cross-platform (Windows, macOS, Linux)

## Troubleshooting

### Common Issues

1. **YOLO model download fails:**
   - Ensure internet connection is stable
   - Check if `yolov8n.pt` exists in the project directory
   - Try downloading manually from Ultralytics

2. **GPU not detected:**
   - Install CUDA and cuDNN if using NVIDIA GPU
   - Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
   - System will automatically fall back to CPU

3. **Camera not found:**
   - Check camera index (try 0, 1, 2...)
   - Ensure camera is not used by another application
   - Use `--test-camera 0` to test camera capabilities

4. **Low performance:**
   - Use YOLOv8n (nano) model
   - Reduce camera resolution
   - Lower confidence threshold
   - Enable GPU acceleration

5. **Poor detection quality:**
   - Increase confidence threshold
   - Use larger YOLO model
   - Ensure good lighting conditions
   - Adjust IoU threshold

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Test with different input sources and models
4. Check hardware compatibility (GPU/CUDA)

## Examples

### Basic Usage
```bash
# Webcam detection with default settings
python person_detection_main.py --camera 0

# Process the included sample video
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Process sample video and save output
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/my_detection.mp4"

# High confidence detection
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --conf 0.6 --output "output/high_conf.mp4"
```

### Advanced Usage
```bash
# Test camera capabilities first
python person_detection_main.py --test-camera 0

# HD webcam with custom settings
python person_detection_main.py --camera 0 --width 1920 --height 1080 --no-flip

# Use different YOLO model
python person_detection_main.py --camera 0 --model yolov8s.pt

# Batch processing without display
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/batch_result.mp4" --no-display

# Custom thresholds and scaling
python person_detection_main.py --camera 0 --conf 0.4 --iou 0.5 --scale 0.5
```

### Camera Resolution Testing
```bash
# Test what resolutions your camera supports
python person_detection_main.py --test-camera 0

# Use specific camera resolution
python person_detection_main.py --camera 0 --width 1280 --height 720
```

## Model Performance Comparison

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| YOLOv8n | 6.2MB | ~50ms | ~1.5ms | Good |
| YOLOv8s | 21.5MB | ~80ms | ~2.5ms | Better |
| YOLOv8m | 49.7MB | ~120ms | ~4ms | Very Good |
| YOLOv8l | 83.7MB | ~180ms | ~6ms | Excellent |
| YOLOv8x | 136MB | ~250ms | ~8ms | Best |

*Performance measured on RTX 3080, times are approximate and vary by hardware*

## License

This project uses:
- **Ultralytics YOLOv8**: AGPL-3.0 License
- **OpenCV**: Apache 2.0 License
- **PyTorch**: BSD License
