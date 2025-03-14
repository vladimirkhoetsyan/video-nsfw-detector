# NSFW Video Content Detector

A command-line tool for detecting NSFW content in videos using computer vision techniques. The tool analyzes video frames for potentially inappropriate content and provides frame-by-frame scoring.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vladimirkhoetsyan/video-nsfw-detector.git
cd video-nsfw-detector
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python3 -m nsfw_detector.cli --video <path_to_video> --output <path_to_output.json> [--min-consecutive 3] [--threshold 0.6] [--debug]
```

Options:
- `--video`: Path to the video file to analyze (required)
- `--output`: Path where to save the JSON report (required)
- `--min-consecutive`: Minimum number of consecutive unsafe frames to consider video unsafe (default: 3)
- `--threshold`: Score threshold for unsafe content classification (default: 0.6)
- `--debug`: Enable debug mode to include detailed frame scores in output

Example:
```bash
# Regular analysis
python3 -m nsfw_detector.cli --video ~/Videos/input.mp4 --output ~/results.json --min-consecutive 5 --threshold 0.7

# Analysis with debug information
python3 -m nsfw_detector.cli --video ~/Videos/input.mp4 --output ~/results.json --debug
```

### Output Format

The tool provides two types of output:

#### Console Output
Simple summary showing:
1. Progress bar for frame extraction
2. Progress bar for frame analysis
3. Total frames analyzed
4. Final verdict (SAFE/UNSAFE)
5. Path to the detailed JSON report

#### JSON Report
Detailed analysis in JSON format containing:
```json
{
  "video_path": "path/to/video",
  "total_frames": 1234,
  "is_unsafe": true,
  "unsafe_frames": [
    {
      "frame": 123,
      "timestamp": 4.1,
      "score": 0.87
    }
  ],
  "unsafe_count": 1,
  "unsafe_percentage": 0.08,
  "threshold_used": 0.6,
  "consecutive_unsafe_sequences": [
    {
      "start_frame": 123,
      "end_frame": 125,
      "start_time": 4.1,
      "end_time": 4.2,
      "frame_count": 3,
      "max_score": 0.87
    }
  ],
  "debug": {  // Only present in debug mode
    "frame_scores": [
      {
        "frame": 1,
        "timestamp": 0.0,
        "score": 0.32
      },
      {
        "frame": 2,
        "timestamp": 0.033,
        "score": 0.45
      }
      // ... scores for all frames
    ]
  }
}
```

Key fields in the JSON report:
- `is_unsafe`: Boolean indicating if the video contains NSFW content (based on consecutive unsafe frames)
- `unsafe_frames`: List of all frames that exceeded the threshold
- `consecutive_unsafe_sequences`: List of sequences where multiple consecutive frames exceeded the threshold
- `debug.frame_scores`: (Debug mode only) Complete list of scores for all frames

### Detection Logic

The tool uses two levels of detection:
1. Individual frame analysis: Each frame is scored from 0.3 to 0.9
2. Consecutive frame analysis: To reduce false positives, a video is only marked as UNSAFE if it contains a sequence of consecutive frames above the threshold (configurable via `--threshold`, default: 0.6)

## Customizing Detection Parameters

You can customize the detection behavior in two ways:

1. Using command-line arguments (recommended):
   - `--threshold`: Adjust the score threshold for unsafe content (default: 0.6)
   - `--min-consecutive`: Set minimum consecutive unsafe frames required (default: 3)

2. By modifying the code (advanced):

### 1. Skin Detection Thresholds (`nsfw_detector/core/analyzer.py`)

```python
class NSFWAnalyzer:
    def __init__(self):
        # YCrCb color space thresholds
        self.skin_ycrcb_min = np.array([0, 135, 85], np.uint8)
        self.skin_ycrcb_max = np.array([255, 180, 135], np.uint8)
        
        # HSV color space thresholds
        self.skin_hsv_min = np.array([0, 15, 0], np.uint8)
        self.skin_hsv_max = np.array([17, 170, 255], np.uint8)
        
        # Lab color space thresholds
        self.skin_lab_min = np.array([20, 130, 130], np.uint8)
        self.skin_lab_max = np.array([250, 180, 180], np.uint8)
```

### 2. Content Classification Thresholds

The tool uses the following thresholds for classification:
- Scores > 0.6: Unsafe content
- Scores between 0.5 and 0.6: Questionable content
- Scores < 0.5: Safe content

To adjust these thresholds, modify the score calculation in `analyze_frame()`:

```python
# In nsfw_detector/core/analyzer.py
def analyze_frame(self, frame_path: str) -> float:
    # ... existing code ...
    
    # Adjust these values to change sensitivity
    base_threshold = 0.5     # Threshold for questionable content
    strict_threshold = 0.6   # Default threshold for unsafe content
```

### 3. Frame Extraction Parameters

To modify frame extraction settings, edit `nsfw_detector/core/frame_extractor.py`:

```python
class FrameExtractor:
    def extract_frames(self, video_path: str) -> str:
        # Modify frame output format
        output_path = output_dir / f"frame_{frame_count:04d}.jpg"
        
        # Adjust JPEG quality (0-100)
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
```

## Contributing

Feel free to submit issues and enhancement requests! 