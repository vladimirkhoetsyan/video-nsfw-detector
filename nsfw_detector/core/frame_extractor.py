import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

class FrameExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._fps = None

    def get_fps(self) -> float:
        """Get the FPS of the last processed video."""
        return self._fps if self._fps is not None else 30.0

    def extract_frames(self, video_path: str) -> str:
        """Extract frames from a video file."""
        # Create output directory
        video_id = Path(video_path).stem
        output_dir = Path('data/tmp') / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames with progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            output_path = output_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            pbar.update(1)

        pbar.close()
        cap.release()

        return str(output_dir) 