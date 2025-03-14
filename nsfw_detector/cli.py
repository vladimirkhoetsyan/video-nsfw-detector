import os
import sys
import json
import logging
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from termcolor import colored
from .core.detector import NSFWDetector

def find_consecutive_unsafe_frames(frames, threshold=0.6, min_consecutive=3):
    """Find sequences of consecutive unsafe frames."""
    consecutive_sequences = []
    current_sequence = []
    
    for frame in frames:
        if frame['score'] > threshold:  # Use provided threshold
            current_sequence.append(frame)
        else:
            if len(current_sequence) >= min_consecutive:
                consecutive_sequences.append(current_sequence)
            current_sequence = []
    
    # Check the last sequence
    if len(current_sequence) >= min_consecutive:
        consecutive_sequences.append(current_sequence)
    
    return consecutive_sequences

def main():
    """Main entry point for the NSFW detector CLI."""
    # Set up logging - only show WARNING and above to suppress INFO messages
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Detect NSFW content in videos')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--min-consecutive', type=int, default=3, 
                      help='Minimum number of consecutive unsafe frames to consider video unsafe (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='Score threshold for unsafe content classification (default: 0.6)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode to include detailed frame scores in output')
    args = parser.parse_args()

    try:
        # Initialize detector
        detector = NSFWDetector()
        
        # Analyze video (progress bars will be shown by detector)
        print("\nAnalyzing video for NSFW content...")
        results = detector.analyze_video(args.video)
        
        # Find sequences of consecutive unsafe frames
        consecutive_sequences = find_consecutive_unsafe_frames(results['frames'], args.threshold, args.min_consecutive)
        
        # Get all unsafe frames (for reporting)
        unsafe_frames = []
        total_frames = len(results['frames'])
        unsafe_count = 0
        
        for frame in results['frames']:
            if frame['score'] > args.threshold:  # Use provided threshold
                unsafe_frames.append({
                    'frame': frame['frame'],
                    'timestamp': frame['timestamp'],
                    'score': frame['score']
                })
                unsafe_count += 1
        
        # Determine if video is unsafe (has enough consecutive unsafe frames)
        is_unsafe = len(consecutive_sequences) > 0
        
        # Create simplified report
        simplified_report = {
            'video_path': args.video,
            'total_frames': total_frames,
            'is_unsafe': is_unsafe,
            'unsafe_frames': unsafe_frames,
            'unsafe_count': unsafe_count,
            'unsafe_percentage': (unsafe_count / total_frames) * 100,
            'threshold_used': args.threshold,
            'consecutive_unsafe_sequences': [
                {
                    'start_frame': seq[0]['frame'],
                    'end_frame': seq[-1]['frame'],
                    'start_time': seq[0]['timestamp'],
                    'end_time': seq[-1]['timestamp'],
                    'frame_count': len(seq),
                    'max_score': max(f['score'] for f in seq)
                }
                for seq in consecutive_sequences
            ]
        }

        # Add detailed frame scores in debug mode
        if args.debug:
            simplified_report['debug'] = {
                'frame_scores': [
                    {
                        'frame': frame['frame'],
                        'timestamp': frame['timestamp'],
                        'score': frame['score']
                    }
                    for frame in results['frames']
                ]
            }
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(simplified_report, f, indent=2)
        
        # Print simplified summary with colors
        print("\n" + "="*50)
        print("Analysis Summary:")
        print(f"Total frames analyzed: {total_frames}")
        
        if is_unsafe:
            print(f"\nVerdict: {colored('UNSAFE', 'red', attrs=['bold'])}")
        else:
            print(f"\nVerdict: {colored('SAFE', 'green', attrs=['bold'])}")

        print(f"\nDetailed report saved to: {args.output}")
        if args.debug:
            print("Debug mode: Detailed frame scores included in report")
        
        print("="*50)
            
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        tmp_dir = Path('data/tmp')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    main() 