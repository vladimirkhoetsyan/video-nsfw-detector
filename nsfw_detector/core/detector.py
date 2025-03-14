import os
import json
import logging
from typing import Dict, Any
from tqdm import tqdm
from .frame_extractor import FrameExtractor
from .analyzer import NSFWAnalyzer

class NSFWDetector:
    def __init__(self):
        self.analyzer = NSFWAnalyzer()
        self.frame_extractor = FrameExtractor()

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video file for NSFW content.
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            Dictionary containing analysis results with frame scores
        """
        # Extract frames
        frames_dir = self.frame_extractor.extract_frames(video_path)
        frame_paths = sorted([
            os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
            if f.endswith('.jpg')
        ])
        
        # Analyze frames with progress bar
        frames = []
        pbar = tqdm(total=len(frame_paths), desc="Analyzing frames", unit="frame")
        
        for i, frame_path in enumerate(frame_paths, 1):
            score = self.analyzer.analyze_frame(frame_path)
            frames.append({
                "frame": i,
                "timestamp": i / self.frame_extractor.get_fps(),
                "score": float(score)
            })
            pbar.update(1)
        
        pbar.close()
        
        return {"frames": frames}

    def save_results(self, results: Dict[str, Any], output_path: str):
        self.logger.info(f"Saving analysis results to: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info("Results saved successfully")

    def cleanup(self):
        """Clean up any temporary files"""
        pass 