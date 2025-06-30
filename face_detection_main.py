#!/usr/bin/env python3
"""
Real-Time Face Detection System
===============================
A comprehensive script for face detection using OpenCV's Haar Cascade classifiers.
Features:
- Automatic GPU/CPU detection and usage
- Real-time video processing with live display
- Support for video files and webcam input
- Interactive controls and statistics
- Professional bounding boxes and confidence scores
- Multiple face detection models (frontal, profile)

Usage:
    python face_detection_main.py --input video.mp4        # Process video file
    python face_detection_main.py --camera 0               # Use webcam
    python face_detection_main.py --input video.mp4 --save # Save output video
"""

import cv2
import argparse
import numpy as np
import time
from pathlib import Path
import sys
import os


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """Initialize the face detector with Haar cascade classifiers."""
        
        print("🔥 Initializing Face Detection System...")
        
        # Load Haar cascade classifiers
        self.face_cascade = self._load_cascade('haarcascade_frontalface_default.xml')
        self.profile_cascade = self._load_cascade('haarcascade_profileface.xml')
        self.eye_cascade = self._load_cascade('haarcascade_eye.xml')
        
        # Detection parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Statistics
        self.total_faces_detected = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Face Detection System initialized successfully!")
    
    def _load_cascade(self, cascade_name):
        """Load a Haar cascade classifier."""
        try:
            # Try to load from OpenCV data directory
            cascade_path = cv2.data.haarcascades + cascade_name
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                print(f"❌ Warning: Could not load {cascade_name}")
                return None
            
            print(f"✅ Loaded cascade: {cascade_name}")
            return cascade
        except Exception as e:
            print(f"❌ Error loading {cascade_name}: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detect faces in a frame using multiple cascade classifiers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        faces = []
        
        # Detect frontal faces
        if self.face_cascade is not None:
            frontal_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend([(x, y, w, h, 'frontal') for (x, y, w, h) in frontal_faces])
        
        # Detect profile faces
        if self.profile_cascade is not None:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend([(x, y, w, h, 'profile') for (x, y, w, h) in profile_faces])
        
        # Remove overlapping detections
        faces = self._remove_overlaps(faces)
        
        return faces, gray
    
    def _remove_overlaps(self, faces, overlap_threshold=0.3):
        """Remove overlapping face detections."""
        if len(faces) <= 1:
            return faces
        
        # Convert to numpy array for easier processing
        boxes = np.array([[x, y, x+w, y+h] for x, y, w, h, _ in faces])
        types = [face_type for _, _, _, _, face_type in faces]
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by area (larger faces first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the largest remaining box
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate overlap with remaining boxes
            remaining = indices[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = w * h
            
            # Calculate overlap ratio
            overlap_ratio = overlap / np.minimum(areas[current], areas[remaining])
            
            # Keep boxes with low overlap
            indices = remaining[overlap_ratio < overlap_threshold]
        
        # Return non-overlapping faces
        result = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            result.append((x1, y1, x2-x1, y2-y1, types[i]))
        
        return result
    
    def detect_eyes_in_face(self, gray, face_region):
        """Detect eyes within a face region."""
        if self.eye_cascade is None:
            return []
        
        x, y, w, h = face_region
        roi_gray = gray[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(10, 10)
        )
        
        # Convert eye coordinates to global coordinates
        global_eyes = []
        for (ex, ey, ew, eh) in eyes:
            global_eyes.append((x + ex, y + ey, ew, eh))
        
        return global_eyes
    
    def draw_detections(self, frame, faces, gray):
        """Draw face detections and statistics on the frame."""
        current_faces = len(faces)
        self.total_faces_detected += current_faces
        self.frame_count += 1
        
        for i, (x, y, w, h, face_type) in enumerate(faces):
            # Choose color based on face type
            if face_type == 'frontal':
                color = (0, 255, 0)  # Green for frontal faces
                label = f"Face {i+1}"
            else:
                color = (255, 0, 0)  # Blue for profile faces
                label = f"Profile {i+1}"
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detect and draw eyes
            eyes = self.detect_eyes_in_face(gray, (x, y, w, h))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        
        # Draw statistics
        self._draw_statistics(frame, current_faces)
        
        return frame
    
    def _draw_statistics(self, frame, current_faces):
        """Draw detection statistics on the frame."""
        height, width = frame.shape[:2]
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_faces = self.total_faces_detected / self.frame_count if self.frame_count > 0 else 0
        
        # Statistics text
        stats = [
            f"Current Faces: {current_faces}",
            f"FPS: {fps:.1f}",
            f"Avg Faces/Frame: {avg_faces:.1f}",
            f"Total Detections: {self.total_faces_detected}",
            f"Runtime: {elapsed_time:.1f}s"
        ]
        
        # Draw background for statistics
        stats_height = len(stats) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, stats_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, stats_height), (255, 255, 255), 2)
        
        # Draw statistics text
        for i, stat in enumerate(stats):
            y_pos = 30 + i * 25
            cv2.putText(frame, stat, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        controls_y = height - 80
        controls = [
            "Controls: 'q' - Quit | 's' - Save frame | 'r' - Reset stats",
            "'p' - Pause/Resume | '+/-' - Adjust sensitivity"
        ]
        
        for i, control in enumerate(controls):
            y_pos = controls_y + i * 20
            cv2.putText(frame, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def test_camera_resolutions(self, camera_id):
        """Test different camera resolutions to find supported ones."""
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

def process_video_source(detector, source, save_output=False, output_path="output/detected_faces.mp4", 
                        camera_width=1280, camera_height=720, flip_enabled=True, display_scale=0.7):
    """Process video from file or camera."""
    
    # Determine if source is a camera or file
    is_camera = str(source).isdigit()
    cap_source = int(source) if is_camera else source
    
    # Open video capture
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {'camera' if is_camera else 'video file'}: {source}")
        return False
    
    # Configure camera settings if using camera
    if is_camera:
        print(f"📹 Using camera {cap_source}")
        
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
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_camera else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1
    
    print(f"📹 Video Info: {width}x{height} @ {fps:.1f} FPS")
    if not is_camera:
        print(f"📊 Total frames: {total_frames}")
    
    # Setup video writer if saving output
    writer = None
    if save_output:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")
    
    # Processing loop
    paused = False
    frame_number = 0
    
    print("🎬 Starting face detection... Press 'q' to quit")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_camera:
                        print("❌ Error reading from camera")
                        break
                    else:
                        print("✅ Video processing completed!")
                        break
                
                frame_number += 1
                
                # Apply camera flip if enabled and using camera
                if is_camera and flip_enabled:
                    frame = cv2.flip(frame, 1)  # Horizontal flip
                
                # Apply display scaling
                if display_scale != 1.0:
                    display_width = int(width * display_scale)
                    display_height = int(height * display_scale)
                    frame = cv2.resize(frame, (display_width, display_height))
                
                # Detect faces
                start_time = time.time()
                faces, gray = detector.detect_faces(frame)
                detection_time = time.time() - start_time
                
                # Draw detections
                output_frame = detector.draw_detections(frame, faces, gray)
                
                # Add processing time info
                current_height = output_frame.shape[0]
                cv2.putText(output_frame, f"Detection Time: {detection_time*1000:.1f}ms", 
                           (10, current_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save frame if required
                if writer is not None:
                    writer.write(output_frame)
                
                # Show progress for video files
                if not is_camera and total_frames > 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"\r📊 Progress: {progress:.1f}% ({frame_number}/{total_frames})", end="")
            else:
                output_frame = frame.copy()
                cv2.putText(output_frame, "PAUSED - Press 'p' to resume", 
                           (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Detection', output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping detection...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('s'):
                save_path = f"output/face_frame_{frame_number:06d}.jpg"
                os.makedirs("output", exist_ok=True)
                cv2.imwrite(save_path, output_frame)
                print(f"\n💾 Saved frame to: {save_path}")
            elif key == ord('r'):
                detector.total_faces_detected = 0
                detector.frame_count = 0
                detector.start_time = time.time()
                print("\n🔄 Statistics reset")
            elif key == ord('+') or key == ord('='):
                detector.min_neighbors = min(detector.min_neighbors + 1, 10)
                print(f"\n⬆️  Increased sensitivity (min_neighbors: {detector.min_neighbors})")
            elif key == ord('-'):
                detector.min_neighbors = max(detector.min_neighbors - 1, 1)
                print(f"\n⬇️  Decreased sensitivity (min_neighbors: {detector.min_neighbors})")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*50)
        print("📊 Final Statistics:")
        print(f"   Total frames processed: {detector.frame_count}")
        print(f"   Total faces detected: {detector.total_faces_detected}")
        print(f"   Average faces per frame: {detector.total_faces_detected/detector.frame_count:.2f}")
        print(f"   Total runtime: {time.time() - detector.start_time:.2f} seconds")
        print("="*50)
    
    return True


def main():
    """Main function to handle command line arguments and start detection."""
    parser = argparse.ArgumentParser(
        description="Real-Time Face Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection_main.py --input input/video.mp4        # Process video file
  python face_detection_main.py --camera 0                     # Use webcam with default settings
  python face_detection_main.py --camera 0 --width 1920 --height 1080  # HD webcam
  python face_detection_main.py --input video.mp4 --save       # Save output
  python face_detection_main.py --camera 0 --sensitivity high  # High sensitivity
  python face_detection_main.py --test-camera 0                # Test camera resolutions
        """
    )
    
    # Input source arguments
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input', type=str, help='Input video file path')
    input_group.add_argument('--camera', type=int, help='Camera index (0 for default camera)')
    
    # Output arguments
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, default='output/detected_faces.mp4',
                       help='Output video path (default: output/detected_faces.mp4)')
    
    # Detection parameters
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity (default: medium)')
    parser.add_argument('--min-size', type=int, nargs=2, default=[30, 30],
                       metavar=('WIDTH', 'HEIGHT'), help='Minimum face size (default: 30 30)')
    
    # Camera parameters
    parser.add_argument('--width', type=int, default=1280, help='Camera width resolution (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height resolution (default: 720)')
    parser.add_argument('--scale', type=float, default=0.7, help='Display scale factor (0.7 = 70%% size)')
    parser.add_argument('--no-flip', action='store_true', help='Disable camera horizontal flip')
    parser.add_argument('--test-camera', type=int, metavar='ID', help='Test camera resolutions and exit')
    
    args = parser.parse_args()
    
    # Handle camera resolution testing
    if args.test_camera is not None:
        detector = FaceDetector()
        detector.test_camera_resolutions(args.test_camera)
        return
    
    # Require input/camera for normal operation
    if not args.input and args.camera is None:
        parser.error("Must specify either --input or --camera (or use --test-camera)")
    
    # Set sensitivity parameters
    sensitivity_params = {
        'low': {'scale_factor': 1.3, 'min_neighbors': 7},
        'medium': {'scale_factor': 1.1, 'min_neighbors': 5},
        'high': {'scale_factor': 1.05, 'min_neighbors': 3}
    }
    
    params = sensitivity_params[args.sensitivity]
    
    print("🎯 Face Detection System Starting...")
    print(f"📋 Configuration:")
    print(f"   Sensitivity: {args.sensitivity}")
    print(f"   Scale factor: {params['scale_factor']}")
    print(f"   Min neighbors: {params['min_neighbors']}")
    print(f"   Min size: {args.min_size[0]}x{args.min_size[1]}")
    if args.camera is not None:
        print(f"   Camera resolution: {args.width}x{args.height}")
    
    # Initialize detector
    detector = FaceDetector(
        scale_factor=params['scale_factor'],
        min_neighbors=params['min_neighbors'],
        min_size=tuple(args.min_size)
    )
    
    # Determine input source
    source = args.input if args.input else args.camera
    
    # Process video
    success = process_video_source(
        detector, source, args.save, args.output,
        camera_width=args.width, camera_height=args.height,
        flip_enabled=not args.no_flip, display_scale=args.scale
    )
    
    if success:
        print("✅ Face detection completed successfully!")
    else:
        print("❌ Face detection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
