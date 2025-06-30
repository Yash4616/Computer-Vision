#!/usr/bin/env python3
"""
Real-Time Person Detection with YOLO
====================================
A unified script for person detection and head counting using YOLOv8.
Features:
- Automatic GPU/CPU detection and usage
- Real-time video processing with live display
- Support for video files and webcam input
- Interactive controls and statistics
- Professional bounding boxes and confidence scores

Usage:
    python person_detection_main.py --input video.mp4        # Process video file
    python person_detection_main.py --camera 0               # Use webcam
    python person_detection_main.py --input video.mp4 --save # Save output video
"""

import cv2
import torch
import argparse
import numpy as np
import time
from pathlib import Path
import sys

# Set safe globals for YOLOv8 model loading (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.3, iou_threshold=0.45):
        """Initialize the person detector with automatic GPU/CPU selection."""
        
        print("🔥 Initializing Person Detection System...")
        
        # Detect and set device (GPU/CPU)
        self.device = self._get_best_device()
        print(f"🖥️  Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Detection parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.fps_history = []
        self.max_persons = 0
        
        # Colors for bounding boxes
        self.colors = {
            'high_conf': (0, 255, 0),    # Green for high confidence (>0.7)
            'medium_conf': (0, 255, 255), # Yellow for medium confidence (0.5-0.7)
            'low_conf': (0, 165, 255),   # Orange for low confidence (<0.5)
            'text_bg': (0, 0, 0),        # Black background for text
            'text': (255, 255, 255)      # White text
        }
        
        print("✅ Person Detection System Ready!")
    
    def _get_best_device(self):
        """Automatically detect and return the best available device."""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device_id = torch.cuda.current_device()
                device = f"cuda:{device_id}"
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                return device
            else:
                print("💻 GPU not available, using CPU")
                print("💡 Install CUDA-enabled PyTorch for GPU acceleration:")
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                return "cpu"
        except Exception as e:
            print(f"⚠️  GPU detection error: {e}")
            print("💻 Falling back to CPU")
            return "cpu"
    
    def _load_model(self, model_path):
        """Load and configure the YOLO model."""
        try:
            print(f"📥 Loading YOLO model: {model_path}")
            
            # Handle PyTorch 2.6+ security restrictions
            import torch._C
            original_load = torch.load
            
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = safe_load
            
            model = YOLO(model_path)
            
            # Restore original load function
            torch.load = original_load
            
            # Move model to device
            model.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Downloading YOLOv8n model...")
            try:
                # Handle PyTorch security for downloaded model too
                import torch._C
                original_load = torch.load
                
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                
                model = YOLO('yolov8n.pt')  # This will download if not present
                
                # Restore original load function
                torch.load = original_load
                
                model.to(self.device)
                return model
            except Exception as e2:
                print(f"❌ Failed to download model: {e2}")
                sys.exit(1)
    
    def detect_persons(self, frame):
        """Run person detection on a frame and return results."""
        try:
            # Run inference
            results = self.model(frame, 
                               conf=self.conf_threshold,
                               iou=self.iou_threshold,
                               classes=[0],  # Person class only
                               device=self.device,
                               verbose=False)
            
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class': 'person'
                    })
            
            # Debug: Print detection count occasionally
            if self.total_frames % 100 == 0 and len(detections) > 0:
                print(f"🔍 Debug: Found {len(detections)} persons in frame {self.total_frames}")
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and information on the frame."""
        annotated_frame = frame.copy()
        person_count = len(detections)
        
        # Update statistics
        self.total_detections += person_count
        self.max_persons = max(self.max_persons, person_count)
        
        # Draw each detection
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Choose color based on confidence
            if confidence > 0.7:
                color = self.colors['high_conf']
            elif confidence > 0.5:
                color = self.colors['medium_conf']
            else:
                color = self.colors['low_conf']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare text
            label = f"Person {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw text background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         self.colors['text_bg'], -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return annotated_frame, person_count
    
    def draw_info_panel(self, frame, person_count, fps, frame_number):
        """Draw information panel on the frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Information text
        info_lines = [
            f"Person Count: {person_count}",
            f"FPS: {fps:.1f}",
            f"Frame: {frame_number}",
            f"Max Persons: {self.max_persons}",
            f"Device: {self.device.upper()}"
        ]
        
        # Draw each line
        for i, line in enumerate(info_lines):
            y_pos = 35 + (i * 20)
            cv2.putText(frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_controls_help(self, frame):
        """Draw control instructions on the frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 80), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Control instructions
        controls = "Controls: [Q]uit | [P]ause | [S]ave | [H]ide Info | [F]lip | [+/-] Zoom"
        cv2.putText(frame, controls, (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self, input_source, output_path=None, show_display=True, 
                     camera_width=1920, camera_height=1080, flip_enabled=True, display_scale=0.7):
        """Process video with real-time detection and display."""
        
        # Initialize video capture
        if isinstance(input_source, int):  # Camera
            cap = cv2.VideoCapture(input_source)
            print(f"📹 Using camera {input_source}")
            
            # Configure camera settings for better quality
            if cap.isOpened():
                # Try to set the requested resolution first
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Check what resolution we actually got
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # If the camera doesn't support the requested resolution, try common ones
                if actual_width != camera_width or actual_height != camera_height:
                    print(f"⚠️  Requested {camera_width}x{camera_height} not supported, got {actual_width}x{actual_height}")
                    print("🔧 Trying alternative resolutions...")
                    
                    # Try common resolutions in order of preference
                    common_resolutions = [
                        (1280, 720),   # 720p HD
                        (1920, 1080),  # 1080p Full HD  
                        (640, 480),    # VGA
                        (800, 600),    # SVGA
                        (1024, 768),   # XGA
                    ]
                    
                    best_resolution = None
                    for width, height in common_resolutions:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        test_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        test_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        if test_width == width and test_height == height:
                            best_resolution = (width, height)
                            print(f"✅ Using resolution: {width}x{height}")
                            break
                    
                    if not best_resolution:
                        # Use whatever the camera gives us
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"⚠️  Using camera default: {actual_width}x{actual_height}")
                
                # Enable auto-exposure and auto-focus
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable auto focus
                
                # Get final resolution
                final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                final_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"🔧 Camera configured: {final_width}x{final_height} @ {final_fps:.1f}fps")
        else:  # Video file
            if not Path(input_source).exists():
                print(f"❌ Video file not found: {input_source}")
                return
            cap = cv2.VideoCapture(str(input_source))
            print(f"🎥 Processing video: {input_source}")
        
        if not cap.isOpened():
            print("❌ Failed to open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Initialize video writer if saving
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save to: {output_path}")
        
        # Processing variables
        frame_count = 0
        paused = False
        show_info = True
        flip_mode = 1 if flip_enabled else None  # Start with horizontal flip, None = no flip
        # Flip modes: None=no flip, 1=horizontal, 0=vertical, -1=both
        start_time = time.time()
        
        print("\n🚀 Starting real-time detection...")
        print("📺 Controls: [Q]uit | [P]ause | [H]ide Info | [F]lip Modes | [+/-] Zoom | [0] Reset Zoom")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("📋 End of video or failed to read frame")
                        break
                    
                    # Fix camera orientation issues
                    if isinstance(input_source, int) and flip_mode is not None:  # Only for camera feed
                        # Apply the selected flip mode
                        frame = cv2.flip(frame, flip_mode)
                    
                    frame_count += 1
                    self.total_frames += 1
                
                # Calculate FPS
                current_time = time.time()
                elapsed = current_time - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:  # Keep last 30 FPS measurements
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                if not paused:
                    # Run person detection
                    detections = self.detect_persons(frame)
                    
                    # Draw detections
                    annotated_frame, person_count = self.draw_detections(frame, detections)
                else:
                    annotated_frame = frame
                    person_count = 0
                
                # Add information overlays
                if show_info:
                    annotated_frame = self.draw_info_panel(annotated_frame, person_count, avg_fps, frame_count)
                    annotated_frame = self.draw_controls_help(annotated_frame)
                
                # Display frame
                if show_display:
                    display_frame = annotated_frame
                    
                    # Apply display scaling if needed
                    if display_scale != 1.0:
                        display_height, display_width = annotated_frame.shape[:2]
                        new_width = int(display_width * display_scale)
                        new_height = int(display_height * display_scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                    
                    window_name = f"Person Detection - {Path(input_source).name if isinstance(input_source, str) else 'Camera'}"
                    cv2.imshow(window_name, display_frame)
                
                # Save frame if output specified
                if out is not None:
                    out.write(annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q') or key == 27:  # Quit
                    break
                elif key == ord('p') or key == ord('P') or key == 32:  # Pause/Play
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('r') or key == ord('R'):  # Reset
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    start_time = time.time()
                    print("🔄 Video reset")
                elif key == ord('s') or key == ord('S'):  # Save frame
                    save_path = f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"💾 Frame saved: {save_path}")
                elif key == ord('h') or key == ord('H'):  # Toggle info
                    show_info = not show_info
                    print(f"ℹ️  Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord('f') or key == ord('F'):  # Toggle camera flip
                    if isinstance(input_source, int):  # Only for camera
                        # Cycle through flip modes: None -> 1 (horizontal) -> 0 (vertical) -> -1 (both) -> None
                        if flip_mode is None:
                            flip_mode = 1
                            print("🔄 Camera flip: Horizontal (Mirror)")
                        elif flip_mode == 1:
                            flip_mode = 0
                            print("🔄 Camera flip: Vertical (Upside Down)")
                        elif flip_mode == 0:
                            flip_mode = -1
                            print("🔄 Camera flip: Both (180° Rotation)")
                        else:  # flip_mode == -1
                            flip_mode = None
                            print("🔄 Camera flip: OFF")
                    else:
                        print("⚠️  Flip toggle only available for camera input")
                elif key == ord('+') or key == ord('='):  # Zoom in
                    display_scale = min(display_scale * 1.1, 3.0)
                    print(f"🔍 Display scale: {display_scale:.1f}x")
                elif key == ord('-') or key == ord('_'):  # Zoom out
                    display_scale = max(display_scale * 0.9, 0.1)
                    print(f"🔍 Display scale: {display_scale:.1f}x")
                elif key == ord('0'):  # Reset scale
                    display_scale = 1.0
                    print("🔍 Display scale: Reset to 1.0x")
                
                # Progress update for video files
                if isinstance(input_source, str) and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"📊 Progress: {progress:.1f}% - Persons: {person_count}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_stats(elapsed)
    
    def _print_final_stats(self, elapsed_time):
        """Print final processing statistics."""
        print("\n" + "="*50)
        print("📊 FINAL STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.total_frames}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {self.total_frames/elapsed_time:.2f}" if elapsed_time > 0 else "N/A")
        print(f"Total detections: {self.total_detections}")
        print(f"Maximum persons in frame: {self.max_persons}")
        print(f"Average persons per frame: {self.total_detections/self.total_frames:.1f}" if self.total_frames > 0 else "N/A")
        print(f"Device used: {self.device}")
        print("="*50)
    
    def test_camera_resolutions(self, camera_id=0):
        """Test and display available camera resolutions."""
        print(f"🔍 Testing camera {camera_id} resolutions...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Common resolutions to test
        test_resolutions = [
            (320, 240),    # QVGA
            (640, 480),    # VGA
            (800, 600),    # SVGA
            (1024, 768),   # XGA
            (1280, 720),   # 720p HD
            (1280, 1024),  # SXGA
            (1600, 1200),  # UXGA
            (1920, 1080),  # 1080p Full HD
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K
        ]
        
        supported_resolutions = []
        
        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))
                print(f"✅ {width}x{height}")
            else:
                print(f"❌ {width}x{height} (got {actual_width}x{actual_height})")
        
        cap.release()
        
        print(f"\n📊 Supported resolutions for camera {camera_id}:")
        for width, height in supported_resolutions:
            print(f"   {width}x{height}")
        
        if supported_resolutions:
            recommended = supported_resolutions[-1]  # Highest resolution
            print(f"\n💡 Recommended: --width {recommended[0]} --height {recommended[1]}")
        
        return supported_resolutions

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Real-Time Person Detection with YOLO")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", type=str, help="Path to input video file")
    input_group.add_argument("--camera", type=int, help="Camera device ID (0, 1, etc.)")
    
    # Detection parameters
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    
    # Camera options
    parser.add_argument("--width", type=int, default=1920, help="Camera width resolution (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Camera height resolution (default: 1080)")
    parser.add_argument("--scale", type=float, default=0.7, help="Display scale factor (0.7 = 70%% size)")
    parser.add_argument("--no-flip", action="store_true", help="Disable camera horizontal flip")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output video path (optional)")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--test-camera", type=int, metavar='ID', help="Test camera resolutions and exit")
    
    args = parser.parse_args()
    
    # Handle camera resolution testing
    if args.test_camera is not None:
        detector = PersonDetector()
        detector.test_camera_resolutions(args.test_camera)
        return 0
    
    # Require input/camera for normal operation
    if not args.input and args.camera is None:
        parser.error("Must specify either --input or --camera (or use --test-camera)")
    
    try:
        # Initialize detector
        detector = PersonDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # Determine input source
        input_source = args.input if args.input else args.camera
        
        # Process video
        detector.process_video(
            input_source=input_source,
            output_path=args.output,
            show_display=not args.no_display,
            camera_width=args.width,
            camera_height=args.height,
            flip_enabled=not args.no_flip,
            display_scale=args.scale
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
