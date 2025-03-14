import logging
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

class NSFWAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_threshold = 0.55  # Base threshold for general detection
        self.strict_threshold = 0.65  # Stricter threshold for sensitive content
        self.skin_ycrcb_min = np.array([0, 135, 85], np.uint8)
        self.skin_ycrcb_max = np.array([255, 180, 135], np.uint8)
        self.skin_hsv_min = np.array([0, 15, 0], np.uint8)
        self.skin_hsv_max = np.array([17, 170, 255], np.uint8)
        self.skin_lab_min = np.array([20, 130, 130], np.uint8)
        self.skin_lab_max = np.array([250, 180, 180], np.uint8)

    def analyze_frame(self, frame_path: str) -> float:
        """
        Analyze a single frame for NSFW content.
        Returns a score between 0.3 and 0.9 where higher values indicate more unsafe content.
        """
        try:
            # Read and preprocess the image
            img = cv2.imread(frame_path)
            if img is None:
                return 0.3  # Return safe score for invalid images
            
            # Convert to different color spaces for better skin detection
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            
            # Skin detection in YCrCb space
            skin_ycrcb = cv2.inRange(img_ycrcb, (0, 133, 77), (255, 173, 127))
            
            # Skin detection in HSV space
            lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
            upper_hsv = np.array([17, 170, 255], dtype=np.uint8)
            skin_hsv1 = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            
            lower_hsv2 = np.array([170, 15, 0], dtype=np.uint8)
            upper_hsv2 = np.array([180, 170, 255], dtype=np.uint8)
            skin_hsv2 = cv2.inRange(img_hsv, lower_hsv2, upper_hsv2)
            
            skin_hsv = cv2.add(skin_hsv1, skin_hsv2)
            
            # Skin detection in Lab space
            lower_lab = np.array([20, 130, 130], dtype=np.uint8)
            upper_lab = np.array([250, 170, 170], dtype=np.uint8)
            skin_lab = cv2.inRange(img_lab, lower_lab, upper_lab)
            
            # Combine skin masks
            skin_mask = cv2.bitwise_and(cv2.bitwise_and(skin_ycrcb, skin_hsv), skin_lab)
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
            skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
            
            # Calculate skin percentage
            total_pixels = skin_mask.size
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_percentage = skin_pixels / total_pixels
            
            # Find and analyze regions
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize variables for region analysis
            large_regions = 0
            intimate_regions = 0
            total_area = img.shape[0] * img.shape[1]
            center_x = img.shape[1] / 2
            center_y = img.shape[0] / 2
            
            for contour in contours:
                area = cv2.contourArea(contour)
                area_ratio = area / total_area
                
                # Skip very small regions
                if area_ratio < 0.015:  # Increased from 0.02
                    continue
                    
                # Get region properties
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Calculate center weight (regions closer to center get higher weight)
                region_center_x = x + w/2
                region_center_y = y + h/2
                distance_from_center = np.sqrt(((region_center_x - center_x) / center_x) ** 2 + 
                                            ((region_center_y - center_y) / center_y) ** 2)
                center_weight = 1 / (1 + distance_from_center)
                
                # Classify regions based on properties
                if 0.3 < aspect_ratio < 2.8:  # Expanded range
                    if area_ratio > 0.1:  # Large regions
                        large_regions += 1 * center_weight
                    if 0.6 < aspect_ratio < 1.6 and 0.05 < area_ratio < 0.3:  # Intimate regions
                        intimate_regions += 1.5 * center_weight  # Increased weight
            
            # Calculate base score from skin percentage
            base_score = min(0.9, skin_percentage * 2.0)  # Reduced multiplier from 2.5
            
            # Adjust score based on regions
            region_score = min(0.9, (large_regions * 0.15 + intimate_regions * 0.25))  # Adjusted weights
            
            # Calculate final score with more weight on region analysis
            final_score = (base_score * 0.4 + region_score * 0.6)  # More weight on region analysis
            
            # Normalize score to 0.3-0.9 range with adjusted thresholds
            normalized_score = 0.3 + (final_score * 0.6)
            
            # Apply thresholds for classification
            base_threshold = 0.5  # Threshold for questionable content
            strict_threshold = 0.86  # Increased threshold for unsafe content
            
            if normalized_score < base_threshold:
                return 0.3  # Safe
            elif normalized_score > strict_threshold:
                return min(0.9, normalized_score)  # Unsafe
            else:
                return normalized_score  # Questionable
                
        except Exception as e:
            logging.error(f"Error analyzing frame {frame_path}: {str(e)}")
            return 0.3  # Return safe score on error
            
    def analyze_frames(self, frame_paths: List[str], batch_size: int = 16, max_workers: int = 4) -> List[Tuple[int, float]]:
        """
        Analyze frames for NSFW content.
        
        Args:
            frame_paths: List of paths to frame images
            batch_size: Number of frames to process in each batch
            max_workers: Maximum number of worker threads
            
        Returns:
            List of tuples containing (frame_number, score)
        """
        self.logger.info("Analyzing frames...")
        
        frame_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a list to store future objects
            futures = []
            
            # Submit all frame analysis tasks
            for i, frame_path in enumerate(frame_paths):
                future = executor.submit(self.analyze_frame, frame_path)
                futures.append((i + 1, future))
                
                if (i + 1) % batch_size == 0:
                    self.logger.info(f"Submitted {i + 1}/{len(frame_paths)} frames for analysis")
            
            # Collect results as they complete
            for frame_num, future in futures:
                score = future.result()
                frame_scores.append((frame_num, float(score)))
                
                if frame_num % batch_size == 0:
                    self.logger.info(f"Processed {frame_num}/{len(frame_paths)} frames")
        
        # Sort results by frame number
        frame_scores.sort(key=lambda x: x[0])
        self.logger.info(f"Completed analysis of {len(frame_scores)} frames")
        return frame_scores

    def is_frame_unsafe(self, score: float) -> bool:
        """
        Determine if a frame is unsafe based on its score.
        Uses a threshold of 0.84 to minimize false positives.
        """
        return score > 0.84  # Threshold for unsafe content classification