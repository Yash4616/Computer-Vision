#!/usr/bin/env python3
"""
Run Vision Detection Metrics Evaluation
"""

from calculate_metrics import MetricsCalculator
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("🎯 VISION DETECTION SYSTEM - METRICS EVALUATION")
    print("="*70)
    
    # Initialize calculator
    calc = MetricsCalculator()
    
    # Define input videos
    video_dir = Path("input")
    videos = [
        "4122569-uhd_3840_2160_24fps.mp4",
        "5181484-uhd_3840_2160_30fps.mp4"
    ]
    
    person_results = {}
    face_results = {}
    
    # Evaluate each video
    for video_name in videos:
        video_path = video_dir / video_name
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📹 Processing: {video_name}")
        print(f"{'='*70}")
        
        # Person detection evaluation
        print("\n🔍 Running Person Detection Evaluation...")
        person_results[video_name] = calc.evaluate_person_detection(video_path, conf_threshold=0.3)
        
        # Face detection evaluation
        print("\n👤 Running Face Detection Evaluation...")
        face_results[video_name] = calc.evaluate_face_detection(video_path)
    
    # Generate comprehensive stats file
    print("\n" + "="*70)
    print("📝 Generating Comprehensive Stats Report...")
    print("="*70)
    
    calc.generate_stats_file(person_results, face_results, "stats.md")
    
    print("\n✅ Metrics evaluation complete!")
    print("📄 Results saved to: stats.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
