# Vision Detection Systems

A comprehensive computer vision toolkit featuring both **Person Detection** (YOLO-based) and **Face Detection** (OpenCV-based) systems.

## Overview

This workspace contains two powerful detection systems:

1. **Person Detection System** (`person_detection_main.py`) - Uses YOLOv8 neural network for accurate person detection
2. **Face Detection System** (`face_detection_main.py`) - Uses OpenCV Haar Cascades for fast face detection

## Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

Install dependencies for both systems:

```bash
# Core dependencies
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0

# For Person Detection (YOLO-based)
pip install ultralytics>=8.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0

# For Face Detection (OpenCV-based)
pip install opencv-contrib-python>=4.8.0

# Additional utilities
pip install filterpy==1.4.5
pip install scikit-image==0.21.0
pip install imutils==0.5.4
```

Or install all at once using the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Person Detection (YOLO-based)

```bash
# Webcam detection
python person_detection_main.py --camera 0

# Process video file
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Save output
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/detected_people.mp4"
```

### Face Detection (OpenCV-based)

```bash
# Webcam detection
python face_detection_main.py --camera 0

# Process video file
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4"

# Save output
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save
```

## System Comparison

| Feature | Person Detection | Face Detection |
|---------|------------------|----------------|
| **Algorithm** | YOLOv8 Neural Network | OpenCV Haar Cascades |
| **Accuracy** | Very High | High |
| **Speed** | Fast (with GPU) | Very Fast |
| **GPU Support** | Yes | No |
| **Model Size** | 6-136MB | Built-in |
| **Detection Types** | Full body persons | Faces + Eyes |
| **Best For** | Person counting, tracking | Face recognition prep |

## Sample Video

Both systems can process the included sample video:
- **File**: `input/1338598-hd_1920_1080_30fps.mp4`
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS

## Test Results

### System Specifications
- **OS**: Windows 11
- **CPU**: Intel/AMD Multi-core processor
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0GB)
- **RAM**: Available system memory
- **Camera**: Integrated webcam (1920x1080 support)

### Face Detection Test Results

**Configuration:**
```
🎯 Face Detection System Starting...
📋 Configuration:
   Sensitivity: medium
   Scale factor: 1.1
   Min neighbors: 5
   Min size: 30x30
```

**Initialization:**
```
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
🎬 Starting face detection...
```

**Performance**: Successfully processed Full HD video at ~30 FPS with real-time face detection

### Person Detection Test Results

**Configuration:**
```
🔥 Initializing Person Detection System...
🚀 GPU detected: NVIDIA GeForce RTX 4050 Laptop GPU (6.0GB)
🖥️  Using device: cuda:0
📥 Loading YOLO model: yolov8n.pt
✅ Model loaded successfully on cuda:0
✅ Person Detection System Ready!
```

**Performance**: GPU-accelerated processing with YOLOv8 nano model

### Camera Capabilities Test

**Face Detection System:**
```
🔍 Testing camera 0 resolutions...
📊 Supported resolutions for camera 0:
   1920x1080
💡 Recommended: --width 1920 --height 1080
```

**Person Detection System:**
```
🔍 Testing camera 0 resolutions...
📊 Supported resolutions for camera 0:
   1920x1080
💡 Recommended: --width 1920 --height 1080
```

### Output Files Generated
- ✅ `output/detected_faces.mp4` - Face detection results
- ✅ `output/detected_people.mp4` - Person detection results

Both systems successfully processed the sample video and generated output files.

## Verification Commands

To verify the systems work on your machine, run these test commands:

### Quick System Check
```bash
# Test all dependencies
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print('YOLO: Available')"
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

# Test camera capabilities
python face_detection_main.py --test-camera 0
python person_detection_main.py --test-camera 0
```

### Live Demo
```bash
# Face detection on webcam (press 'q' to quit)
python face_detection_main.py --camera 0

# Person detection on webcam (press 'q' to quit)  
python person_detection_main.py --camera 0
```

### Sample Video Processing
```bash
# Process sample video with face detection
python face_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --save

# Process sample video with person detection
python person_detection_main.py --input "input/1338598-hd_1920_1080_30fps.mp4" --output "output/my_test.mp4"
```

## File Structure

```
Vision/
├── README.md                       # This file - main documentation
├── README_person_detection.md     # Person detection detailed guide
├── README_face_detection.md       # Face detection detailed guide
├── person_detection_main.py       # YOLO-based person detection
├── face_detection_main.py         # OpenCV-based face detection
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # YOLO model file
├── input/                         # Input video files
│   ├── 1338598-hd_1920_1080_30fps.mp4
│   └── README.txt
└── output/                        # Output files
    ├── detected_people.mp4        # Person detection output
    ├── detected_faces.mp4         # Face detection output
    └── README.txt
```

## Interactive Controls

Both systems support these keyboard shortcuts during operation:

- **'q'** - Quit the application
- **'p'** - Pause/Resume processing
- **'s'** - Save current frame as image
- **'r'** - Reset statistics counters

## Hardware Requirements

### Minimum
- CPU: Modern multi-core processor
- RAM: 4GB
- Storage: 2GB free space

### Recommended
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 8GB
- GPU: NVIDIA GTX 1060 or better (for person detection)
- Storage: 5GB free space

### Optimal
- CPU: Intel i7/AMD Ryzen 7 or better
- RAM: 16GB
- GPU: NVIDIA RTX 30/40 series (for person detection)
- Storage: 10GB free space

## Detailed Documentation

For comprehensive guides, see the individual README files:

- **[Person Detection Guide](README_person_detection.md)** - Complete YOLO-based person detection documentation
- **[Face Detection Guide](README_face_detection.md)** - Complete OpenCV-based face detection documentation

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Camera not found**: Try different camera indices (0, 1, 2...)
3. **Poor performance**: Use appropriate model size and resolution
4. **GPU not detected**: Install CUDA for NVIDIA GPUs (person detection only)

## Performance Tips

1. **Start with face detection** - It's faster and requires no additional models
2. **Use the sample video** - Test both systems with the included HD video
3. **Test camera first** - Use `--test-camera 0` to check camera capabilities
4. **Choose appropriate models** - Use YOLOv8n for speed, YOLOv8x for accuracy

## License

- **Person Detection**: Uses Ultralytics YOLOv8 (AGPL-3.0)
- **Face Detection**: Uses OpenCV (Apache 2.0)
- **Overall Project**: Open source, see individual component licenses
