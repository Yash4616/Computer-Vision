INPUT FOLDER - Video Processing
==============================

Place your video files in this folder for person detection processing.

SUPPORTED VIDEO FORMATS:
========================
- MP4 (.mp4) - Recommended
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

HOW TO USE:
===========

1. Copy your video files to this folder
2. Run: .\run.bat scan
   (This will show you all available videos)

3. Process a video:
   .\run.bat video "input\{Video Here}"

4. Save processed video to output folder:
   .\run.bat video "input\{Video Here}" --output "output\{Video Here}"

TIPS:
=====
- Larger videos will take longer to process
- Use --no-display for faster processing of long videos
- The GPU will automatically be used for maximum speed
- Processed videos maintain original quality with detection overlays
- Press Q to quit processing early if needed

The person detection AI will:
- Draw bounding boxes around detected persons
- Show confidence scores
- Count people in each frame
- Display processing statistics
- Save results with detection overlays
