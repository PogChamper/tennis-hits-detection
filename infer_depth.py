import os
import sys
import torch
import gc  # For garbage collection
from collections import defaultdict
import json
import numpy as np
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pydantic import BaseModel
import math
from scipy.interpolate import interp1d
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator, Union, Set
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RESOLUTION = 960
DEFAULT_INPUT_SIZE = 518
DEFAULT_SKIP_FRAMES = 11 
DEFAULT_BALL_DEPTH_RADIUS = 10
DEFAULT_MAX_HISTORY = 30  # Maximum ball history frames to display
DEFAULT_MIN_ANGLE_CHANGE = 85  # Minimum angle change to detect a hit
DEFAULT_MIN_SPEED_CHANGE = 400  # Minimum speed change to detect a hit

COLORMAP_DEPTH = cv2.COLORMAP_TURBO  # Colormap for depth visualization

# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class DetectionResult(BaseModel):
    """
    Pydantic model for ball detection results.
    
    Contains information about a detected ball in a video frame, including
    position, tracking information, and metadata.
    """
    detection_class: str
    detection_class_id: int
    object_number: int
    track_id_raw: int
    track_id: int
    score: float
    box: Optional[Tuple[int, int, int, int]] = None
    box_rel: Optional[Tuple[float, float, float, float]] = None
    x_frame: Optional[float] = None
    y_frame: Optional[float] = None
    x_court: Optional[float] = None
    y_court: Optional[float] = None
    z_court: Optional[float] = None
    visible_status: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    frame_id: Optional[int] = None
    camera_id: Optional[str] = None
    timestamp: Optional[float] = None

# Custom implementation of read_video_frames
def read_video_frames_generator(
    video_path: str, 
    process_length: int = -1, 
    target_fps: float = -1, 
    max_res: int = -1, 
    skip_frames: int = 0
) -> Generator[Tuple[np.ndarray, int, int, float, bool], None, None]:
    """
    Generator that yields frames from a video file one at a time to save memory.
    
    Args:
        video_path: Path to the video file.
        process_length: Maximum number of frames to process (-1 for all).
        target_fps: Target FPS for output (-1 for original).
        max_res: Maximum resolution for processing (-1 for original).
        skip_frames: Number of frames to skip for depth processing.
        
    Yields:
        Tuple containing:
            - Frame as RGB numpy array
            - Frame index (0-based)
            - Total number of frames to process
            - Target FPS
            - Whether this is a keyframe for depth processing
            
    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened or processed.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video info: {original_width}x{original_height} @ {original_fps}fps, {frame_count} frames")
        
        # Calculate target fps and stride
        fps = original_fps if target_fps <= 0 else target_fps
        stride = max(round(original_fps / fps), 1)
        
        # Calculate target size
        height, width = original_height, original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(height, width)
            height = int(height * scale)
            width = int(width * scale)
            # Ensure even dimensions for video encoding
            height = height - (height % 2)
            width = width - (width % 2)
        
        logger.info(f"Target size: {width}x{height}, Target FPS: {fps}, Stride: {stride}, Skip frames: {skip_frames}")
        
        # Calculate number of frames to read
        if process_length <= 0:
            process_length = frame_count
        else:
            process_length = min(process_length, frame_count)
            
        # Number of frames to capture after stride
        num_frames = min(process_length, 1 + (frame_count - 1) // stride)
        logger.info(f"Will process {num_frames} frames")
        
        # Read frames one at a time
        frame_idx = 0
        yielded_frames = 0
        skipped_count = 0
        
        while cap.isOpened() and yielded_frames < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % stride == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if width != original_width or height != original_height:
                    frame = cv2.resize(frame, (width, height))
                
                # Determine if this is a keyframe for depth inference
                is_keyframe = (skipped_count % (skip_frames + 1)) == 0
                
                yield frame, yielded_frames, num_frames, fps, is_keyframe
                yielded_frames += 1
                skipped_count += 1
                
            frame_idx += 1
            
        cap.release()
        
    except Exception as e:
        logger.exception(f"Error processing video frames: {e}")
        raise RuntimeError(f"Failed to process video: {e}")

@dataclass
class BallInfo3D:
    """
    Stores 3D ball information including position, depth, and speed.
    
    Attributes:
        x: X-coordinate of the ball in the original frame.
        y: Y-coordinate of the ball in the original frame.
        z: Depth (Z-coordinate) of the ball, if available.
        timestamp: Timestamp of the detection, if available.
        speed_3d: 3D speed of the ball in pixels/sec, calculated between frames.
        interpolated: Whether this ball info has interpolated Z value.
    """
    x: float
    y: float
    z: Optional[float] = None
    timestamp: Optional[float] = None
    speed_3d: Optional[float] = None
    interpolated: bool = False
    
    def calculate_speed(self, prev_ball: 'BallInfo3D') -> Optional[float]:
        """
        Calculate 3D speed between this ball and a previous ball detection.
        
        Args:
            prev_ball: Previous ball detection to calculate speed from.
            
        Returns:
            Calculated speed in pixels per second, or None if calculation is not possible.
        """
        if (prev_ball and prev_ball.z is not None and 
            self.z is not None and 
            prev_ball.timestamp is not None and 
            self.timestamp is not None):
            
            # Calculate 3D distance
            dx = self.x - prev_ball.x
            dy = self.y - prev_ball.y
            dz = self.z - prev_ball.z
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Calculate time difference in seconds
            dt = self.timestamp - prev_ball.timestamp
            
            if dt > 0:
                # Speed in pixels per second
                self.speed_3d = distance / dt
                return self.speed_3d
        return None

# Function to interpolate depth values for skipped frames
def interpolate_z_values(
    ball_info_dict: Dict[int, BallInfo3D], 
    all_frame_ids: List[int]
) -> Dict[int, BallInfo3D]:
    """
    Interpolate Z (depth) values for frames between keyframes.
    
    Uses cubic interpolation to estimate depth values for frames where
    depth wasn't directly measured.
    
    Args:
        ball_info_dict: Dictionary mapping frame IDs to BallInfo3D objects.
        all_frame_ids: List of all frame IDs in the sequence.
        
    Returns:
        Updated dictionary with interpolated Z values.
    """
    # Get all frames with ball info in sorted order
    frame_ids_with_info = sorted(ball_info_dict.keys())
    
    if len(frame_ids_with_info) < 2:
        logger.warning("Not enough data to interpolate (need at least 2 frames)")
        return ball_info_dict  # Not enough data to interpolate
    
    try:
        # First, gather existing z values and timestamps
        existing_frame_ids = []
        existing_z_values = []
        existing_timestamps = []
        
        # Use more efficient list collection
        for frame_id in frame_ids_with_info:
            ball_info = ball_info_dict[frame_id]
            if ball_info.z is not None:
                existing_frame_ids.append(frame_id)
                existing_z_values.append(ball_info.z)
                existing_timestamps.append(ball_info.timestamp if ball_info.timestamp is not None else frame_id)
        
        if len(existing_z_values) < 2:
            logger.warning("Not enough Z values to interpolate (need at least 2 valid depths)")
            return ball_info_dict  # Not enough data with valid Z values to interpolate
        
        # Convert to numpy arrays for faster processing
        existing_timestamps = np.array(existing_timestamps)
        existing_z_values = np.array(existing_z_values)
        
        # Create interpolation function - cubic interpolation when possible
        interp_kind = 'cubic' if len(existing_z_values) >= 4 else 'linear'
        z_interp = interp1d(
            existing_timestamps, 
            existing_z_values, 
            kind=interp_kind, 
            bounds_error=False, 
            fill_value=(existing_z_values[0], existing_z_values[-1])
        )
        
        # Find all frames that need interpolation
        frames_to_interpolate = []
        for frame_id in all_frame_ids:
            if frame_id in ball_info_dict and ball_info_dict[frame_id].z is None:
                frames_to_interpolate.append(frame_id)
        
        # If there are no frames to interpolate, return early
        if not frames_to_interpolate:
            return ball_info_dict
            
        # Perform interpolation in batch
        timestamps_to_interpolate = []
        for frame_id in frames_to_interpolate:
            ball_detection = ball_info_dict[frame_id]
            timestamps_to_interpolate.append(
                ball_detection.timestamp if ball_detection.timestamp is not None else frame_id
            )
            
        # Interpolate all Z values at once
        interpolated_z_values = z_interp(timestamps_to_interpolate)
        
        # Update ball info with interpolated values
        for i, frame_id in enumerate(frames_to_interpolate):
            ball_info_dict[frame_id].z = float(interpolated_z_values[i])
            ball_info_dict[frame_id].interpolated = True
        
        return ball_info_dict
        
    except Exception as e:
        logger.error(f"Error during Z-value interpolation: {e}")
        return ball_info_dict  # Return original in case of error

# Function to detect hits from 3D ball trajectory
def detect_hits(
    ball_info_3d_by_frame: Dict[int, BallInfo3D], 
    fps: float = 30.0, 
    min_angle_change: float = DEFAULT_MIN_ANGLE_CHANGE, 
    min_speed_change: float = DEFAULT_MIN_SPEED_CHANGE
) -> List[int]:
    """
    Detect moments when players hit the ball by analyzing 3D trajectory and speed changes.
    
    Args:
        ball_info_3d_by_frame: Dictionary mapping frame IDs to BallInfo3D objects.
        fps: Frames per second of the video.
        min_angle_change: Minimum angle change to detect a hit (degrees).
        min_speed_change: Minimum speed change to detect a hit (pixels/sec).
        
    Returns:
        List of frame IDs where hits are detected.
    """
    hits = []
    frame_ids = sorted(ball_info_3d_by_frame.keys())
    
    if len(frame_ids) < 3:
        logger.warning("Not enough frames to detect hits (need at least 3)")
        return hits
    
    try:
        # Calculate velocity vectors in 3D
        velocity_vectors = []
        
        for i in range(1, len(frame_ids)):
            prev_frame_id = frame_ids[i-1]
            curr_frame_id = frame_ids[i]
            
            prev_ball = ball_info_3d_by_frame[prev_frame_id]
            curr_ball = ball_info_3d_by_frame[curr_frame_id]
            
            # Skip if depth info is missing
            if prev_ball.z is None or curr_ball.z is None:
                continue
            
            # Time difference in seconds
            time_diff = (curr_frame_id - prev_frame_id) / fps
            
            if time_diff <= 0:
                continue
            
            # Calculate 3D velocity vector
            vx = (curr_ball.x - prev_ball.x) / time_diff
            vy = (curr_ball.y - prev_ball.y) / time_diff
            vz = (curr_ball.z - prev_ball.z) / time_diff
            
            velocity_vectors.append((curr_frame_id, vx, vy, vz, curr_ball.speed_3d))
        
        if len(velocity_vectors) < 2:
            logger.warning("Not enough velocity data to detect hits")
            return hits
        
        # Detect significant direction or speed changes
        window_size = min(3, len(velocity_vectors) // 3)  # Adaptive window size
        if window_size == 0:
            window_size = 1
            
        logger.info(f"Using window size of {window_size} for hit detection")
        
        for i in range(window_size, len(velocity_vectors)):
            curr_frame_id, curr_vx, curr_vy, curr_vz, curr_speed = velocity_vectors[i]
            
            # Calculate average of previous velocity vectors
            window_start = max(0, i - window_size)
            prev_vx = sum(v[1] for v in velocity_vectors[window_start:i]) / (i - window_start)
            prev_vy = sum(v[2] for v in velocity_vectors[window_start:i]) / (i - window_start)
            prev_vz = sum(v[3] for v in velocity_vectors[window_start:i]) / (i - window_start)
            prev_speeds = [v[4] for v in velocity_vectors[window_start:i] if v[4] is not None]
            prev_speed = sum(prev_speeds) / len(prev_speeds) if prev_speeds else None
            
            # Calculate angle between 3D velocity vectors
            dot_product = curr_vx * prev_vx + curr_vy * prev_vy + curr_vz * prev_vz
            magnitude_curr = math.sqrt(curr_vx**2 + curr_vy**2 + curr_vz**2)
            magnitude_prev = math.sqrt(prev_vx**2 + prev_vy**2 + prev_vz**2)
            
            if magnitude_curr > 0 and magnitude_prev > 0:
                cos_angle = max(-1, min(1, dot_product / (magnitude_curr * magnitude_prev)))
                angle = math.degrees(math.acos(cos_angle))
                
                # Calculate speed change
                speed_change = None
                if curr_speed is not None and prev_speed is not None:
                    speed_change = abs(curr_speed - prev_speed)
                
                # Detect hits based on angle change or speed change
                hit_detected = False
                hit_reason = []
                
                if angle > min_angle_change:
                    hit_detected = True
                    hit_reason.append(f"angle={angle:.1f}°")
                    
                if speed_change is not None and speed_change > min_speed_change:
                    hit_detected = True
                    hit_reason.append(f"speed change={speed_change:.1f}")
                    
                if hit_detected:
                    logger.debug(f"Hit detected at frame {curr_frame_id}: {', '.join(hit_reason)}")
                    hits.append(curr_frame_id)
        
        return hits
        
    except Exception as e:
        logger.error(f"Error detecting hits: {e}")
        return []  # Return empty list in case of error

# First, let's add a new function to calculate mean depth in a circular area
def get_circular_mean_depth(
    depth_map: np.ndarray, 
    center_x: int, 
    center_y: int, 
    radius: int = 5
) -> Optional[float]:
    """
    Calculate the mean depth value within a circular area around a point.
    
    Args:
        depth_map: The depth map array (2D numpy array)
        center_x: X-coordinate of the circle center
        center_y: Y-coordinate of the circle center
        radius: Radius of the circle in pixels
    
    Returns:
        Mean depth value within the circle, or None if no valid depths are found
    """
    if depth_map is None:
        return None
        
    try:
        height, width = depth_map.shape
        
        # Clamp coordinates to ensure they're within valid range
        center_x = max(0, min(center_x, width - 1))
        center_y = max(0, min(center_y, height - 1))
        
        # Define bounding box
        x_min = max(0, center_x - radius)
        x_max = min(width, center_x + radius + 1)
        y_min = max(0, center_y - radius) 
        y_max = min(height, center_y + radius + 1)
        
        # Use vectorized operations for better performance
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        dist_squared = (x_grid - center_x)**2 + (y_grid - center_y)**2
        mask = dist_squared <= radius**2
        
        # Extract valid depth values
        valid_depths = depth_map[y_min:y_max, x_min:x_max][mask]
        valid_depths = valid_depths[~np.isnan(valid_depths) & np.isfinite(valid_depths)]
        
        # Return mean if we have valid depths, None otherwise
        return float(np.mean(valid_depths)) if len(valid_depths) > 0 else None
    except Exception as e:
        logger.warning(f"Error calculating circular mean depth: {e}")
        return None

def run_depth_inference(
    video_path: str, 
    ball_results_by_frame_id: Dict[int, List[DetectionResult]], 
    orig_dimensions: Tuple[int, int], 
    model: Any, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run depth inference on video frames and analyze ball trajectory in 3D.
    
    Args:
        video_path: Path to the video file.
        ball_results_by_frame_id: Dictionary mapping frame IDs to ball detection results.
        orig_dimensions: Original video dimensions (width, height).
        model: Depth inference model.
        params: Processing parameters.
        
    Returns:
        Dictionary containing depth inference results, including 3D ball information and hit detection.
    """
    logger.info("=" * 50)
    logger.info("PHASE 1: DEPTH INFERENCE WITH FRAME SKIPPING")
    logger.info("=" * 50)
    
    # Extract parameters
    max_len = params.get('max_len', -1)
    target_fps = params.get('target_fps', -1)
    max_res = params.get('max_res', DEFAULT_MAX_RESOLUTION)
    input_size = params.get('input_size', DEFAULT_INPUT_SIZE)
    skip_frames = params.get('skip_frames', DEFAULT_SKIP_FRAMES)
    ball_depth_radius = params.get('ball_depth_radius', DEFAULT_BALL_DEPTH_RADIUS)
    
    orig_width, orig_height = orig_dimensions
    
    # Initialize storage for results
    all_depths = {}  # Store depth maps for frames with ball detections
    all_frame_data = {}  # Store metadata for all frames
    ball_info_3d_by_frame = {}  # Store 3D ball information
    
    previous_ball_info = None
    frame_dimensions = None
    video_fps = None
    total_frames_count = None
    
    # Process video frames
    try:
        # Initialize frame generator
        frame_generator = read_video_frames_generator(
            video_path, max_len, target_fps, max_res, skip_frames
        )
        
        # Get first frame to set up dimensions
        frame_info = next(frame_generator, None)
        if frame_info is None:
            raise RuntimeError("Failed to read first frame from video")
        
        first_frame, first_frame_idx, total_frames, fps, is_keyframe = frame_info
        frame_dimensions = first_frame.shape[:2]  # (height, width)
        video_fps = fps
        total_frames_count = total_frames
        height, width = frame_dimensions
        
        # Store metadata for first frame
        all_frame_data[first_frame_idx] = {'is_keyframe': is_keyframe}
        
        # Process frames with progress bar
        logger.info(f"Running depth inference on {total_frames} frames (processing 1 of every {skip_frames+1} frames)")
        progress_bar = tqdm(total=total_frames, desc="Depth inference")
        
        # Process first frame
        process_frame_depth(
            first_frame, first_frame_idx, is_keyframe, model, input_size,
            ball_results_by_frame_id, orig_dimensions, frame_dimensions, 
            ball_depth_radius, all_depths, ball_info_3d_by_frame
        )
        
        # Update previous ball info
        if first_frame_idx in ball_info_3d_by_frame:
            previous_ball_info = ball_info_3d_by_frame[first_frame_idx]
        
        # Update progress
        progress_bar.update(1)
        
        # Process remaining frames
        for frame, frame_id, _, _, is_keyframe in frame_generator:
            # Store frame metadata
            all_frame_data[frame_id] = {'is_keyframe': is_keyframe}
            
            # Process frame depth - only if it's a keyframe or has ball detections
            if is_keyframe or frame_id in ball_results_by_frame_id:
                process_frame_depth(
                    frame, frame_id, is_keyframe, model, input_size,
                    ball_results_by_frame_id, orig_dimensions, frame_dimensions, 
                    ball_depth_radius, all_depths, ball_info_3d_by_frame
                )
                
                # Calculate ball speed if we have a previous ball with depth
                if frame_id in ball_info_3d_by_frame and previous_ball_info is not None:
                    current_ball = ball_info_3d_by_frame[frame_id]
                    if previous_ball_info.z is not None and current_ball.z is not None:
                        current_ball.calculate_speed(previous_ball_info)
                    previous_ball_info = current_ball
            
            # Update progress
            progress_bar.update(1)
            
            # Free memory periodically but less aggressively (every 40 frames instead of 20)
            if frame_id % 40 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Close progress bar
        progress_bar.close()
        
        # Get all unique frame IDs
        all_frame_ids = sorted(all_frame_data.keys())
        
        # Interpolate Z values for skipped frames
        logger.info("Interpolating depth values for skipped frames...")
        ball_info_3d_by_frame = interpolate_z_values(ball_info_3d_by_frame, all_frame_ids)
        
        # Calculate speeds for all frames after interpolation
        logger.info("Calculating speeds for all frames...")
        calculate_speeds_for_all_frames(ball_info_3d_by_frame)
        
        # Detect ball hits
        logger.info("Detecting ball hits...")
        hit_frames = detect_hits(ball_info_3d_by_frame, fps=video_fps)
        logger.info(f"Detected {len(hit_frames)} hits at frames: {hit_frames}")
        
        # Create result dictionary
        return {
            'all_depths': all_depths,
            'ball_info_3d_by_frame': ball_info_3d_by_frame,
            'hit_frames': hit_frames,
            'frame_dimensions': frame_dimensions,
            'video_fps': video_fps,
            'total_frames': total_frames_count,
            'all_frame_data': all_frame_data
        }
        
    except Exception as e:
        logger.exception(f"Error during depth inference: {e}")
        raise RuntimeError(f"Depth inference failed: {e}")

def process_frame_depth(
    frame: np.ndarray,
    frame_id: int,
    is_keyframe: bool,
    model: Any,
    input_size: int,
    ball_results_by_frame_id: Dict[int, List[DetectionResult]],
    orig_dimensions: Tuple[int, int],
    frame_dimensions: Tuple[int, int],
    ball_depth_radius: int,
    all_depths: Dict[int, np.ndarray],
    ball_info_3d_by_frame: Dict[int, BallInfo3D]
) -> None:
    """
    Process depth for a single frame and extract 3D ball information.
    
    Args:
        frame: The video frame to process.
        frame_id: Frame ID.
        is_keyframe: Whether this is a keyframe for depth processing.
        model: Depth inference model.
        input_size: Input size for depth model.
        ball_results_by_frame_id: Dictionary of ball detections.
        orig_dimensions: Original video dimensions (width, height).
        frame_dimensions: Current frame dimensions (height, width).
        ball_depth_radius: Radius for circular depth calculation.
        all_depths: Dictionary to store computed depth maps (modified in-place).
        ball_info_3d_by_frame: Dictionary to store 3D ball info (modified in-place).
    """
    height, width = frame_dimensions
    orig_width, orig_height = orig_dimensions
    
    # Only compute depth for keyframes or frames with ball detections
    depth = None
    if is_keyframe and (frame_id in ball_results_by_frame_id):
        # Free CUDA memory before inference if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        depth = model.infer_image(frame, input_size=input_size)
    
    # Process ball detections if available for this frame
    if frame_id in ball_results_by_frame_id:
        # Store depth map if computed
        if depth is not None:
            all_depths[frame_id] = depth
        
        # Process each ball detection
        ball_results = ball_results_by_frame_id[frame_id]
        for ball_result in ball_results:
            # Get ball coordinates
            ball_x = int(ball_result.x_frame)
            ball_y = int(ball_result.y_frame)
            timestamp = ball_result.timestamp
            
            # Get normalized coordinates for depth map
            norm_x = min(int(ball_x * width / orig_width), width - 1)
            norm_y = min(int(ball_y * height / orig_height), height - 1)
            
            # Get depth at ball position using circular mean
            ball_z = None
            if depth is not None:
                ball_z = get_circular_mean_depth(depth, norm_x, norm_y, radius=ball_depth_radius)
            
            # Create 3D ball info
            ball_info = BallInfo3D(
                x=ball_x,
                y=ball_y,
                z=ball_z,
                timestamp=timestamp,
                interpolated=(not is_keyframe)
            )
            ball_info_3d_by_frame[frame_id] = ball_info

def calculate_speeds_for_all_frames(ball_info_3d_by_frame: Dict[int, BallInfo3D]) -> None:
    """
    Calculate 3D speeds for all frames with ball information.
    
    Args:
        ball_info_3d_by_frame: Dictionary mapping frame IDs to BallInfo3D objects.
    """
    frame_ids = sorted(ball_info_3d_by_frame.keys())
    prev_ball_info = None
    
    for frame_id in frame_ids:
        ball_info = ball_info_3d_by_frame[frame_id]
        if prev_ball_info is not None and ball_info.z is not None and prev_ball_info.z is not None:
            ball_info.calculate_speed(prev_ball_info)
        prev_ball_info = ball_info

# Function to create the visualization video
def create_visualization_video(
    video_path: str, 
    inference_results: Dict[str, Any], 
    ball_results_by_frame_id: Dict[int, List[DetectionResult]], 
    orig_dimensions: Tuple[int, int], 
    params: Dict[str, Any], 
    output_path: str
) -> str:
    """
    Create a visualization video with 3D ball information and hit detection.
    
    Args:
        video_path: Path to the input video file.
        inference_results: Results from depth inference, including 3D ball info.
        ball_results_by_frame_id: Dictionary of ball detections by frame.
        orig_dimensions: Original video dimensions (width, height).
        params: Processing parameters.
        output_path: Path where the output video will be saved.
        
    Returns:
        Path to the created visualization video.
        
    Raises:
        RuntimeError: If video creation fails.
    """
    logger.info("=" * 50)
    logger.info("PHASE 2: VIDEO VISUALIZATION")
    logger.info("=" * 50)
    
    # Extract parameters
    max_len = params.get('max_len', -1)
    target_fps = params.get('target_fps', -1)
    max_res = params.get('max_res', DEFAULT_MAX_RESOLUTION)
    
    # Extract results from inference
    all_depths = inference_results['all_depths']
    ball_info_3d_by_frame = inference_results['ball_info_3d_by_frame']
    hit_frames = inference_results['hit_frames']
    height, width = inference_results['frame_dimensions']
    video_fps = inference_results['video_fps']
    total_frames = inference_results['total_frames']
    all_frame_data = inference_results['all_frame_data']
    
    orig_width, orig_height = orig_dimensions
    
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
        video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
        
        logger.info(f"Creating visualization video with dimensions {width}x{height} @ {video_fps}fps")
        
        # Initialize state for visualization
        ball_history = []
        max_history = DEFAULT_MAX_HISTORY
        depth_map_cache = {}  # Cache for interpolated depth maps
        
        # Create visualization with hit detection
        frame_generator = read_video_frames_generator(
            video_path, max_len, target_fps, max_res, 0  # No skipping for visualization
        )
        
        # Initialize progress bar for visualization
        progress_bar = tqdm(total=total_frames, desc="Video visualization")
        
        # Process all frames for visualization
        for frame, frame_id, _, _, _ in frame_generator:
            # Create visualization frame
            vis_frame = create_frame_visualization(
                frame, frame_id, all_depths, depth_map_cache, 
                ball_results_by_frame_id, ball_info_3d_by_frame,
                ball_history, hit_frames, orig_dimensions,
                width, height, max_history
            )
            
            # Write frame to video
            video_writer.write(vis_frame)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Free memory periodically
            if frame_id % 20 == 0:
                gc.collect()
        
        # Clean up
        progress_bar.close()
        video_writer.release()
        logger.info(f"Visualization video saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.exception(f"Error creating visualization video: {e}")
        raise RuntimeError(f"Video creation failed: {e}")

def create_frame_visualization(
    frame: np.ndarray,
    frame_id: int,
    all_depths: Dict[int, np.ndarray],
    depth_map_cache: Dict[int, np.ndarray],
    ball_results_by_frame_id: Dict[int, List[DetectionResult]],
    ball_info_3d_by_frame: Dict[int, BallInfo3D],
    ball_history: List[Tuple[float, float]],
    hit_frames: List[int],
    orig_dimensions: Tuple[int, int],
    width: int,
    height: int,
    max_history: int
) -> np.ndarray:
    """
    Create visualization for a single frame.
    
    Args:
        frame: RGB frame to visualize.
        frame_id: Current frame ID.
        all_depths: Dictionary of depth maps.
        depth_map_cache: Cache for interpolated depth maps.
        ball_results_by_frame_id: Dictionary of ball detections.
        ball_info_3d_by_frame: Dictionary of 3D ball information.
        ball_history: List of previous ball positions (modified in-place).
        hit_frames: List of frames where hits are detected.
        orig_dimensions: Original video dimensions.
        width: Current frame width.
        height: Current frame height.
        max_history: Maximum number of historical positions to show.
        
    Returns:
        Visualization frame in BGR format for video writing.
    """
    # Convert frame to BGR for OpenCV once
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    orig_width, orig_height = orig_dimensions
    
    # Skip depth visualization for frames without ball - improves performance
    has_ball = frame_id in ball_results_by_frame_id
    
    # Get depth map only if this frame has a ball detection
    depth_map = None
    if has_ball:
        depth_map = get_frame_depth_map(frame_id, all_depths, depth_map_cache, ball_info_3d_by_frame)
    
    # Visualize depth map if available
    if depth_map is not None:
        blended = visualize_depth_map(depth_map, frame_bgr)
    else:
        blended = frame_bgr.copy()
    
    # Skip the rest of the processing if no ball in this frame
    if not has_ball:
        # Add frame counter
        cv2.putText(
            blended, 
            f"Frame: {frame_id}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
        return blended
    
    # Calculate scaling factors once
    x_scale = width / orig_width
    y_scale = height / orig_height
    
    # Draw ball trajectory history
    for prev_x, prev_y in ball_history:
        cv2.circle(
            blended, 
            (int(prev_x * x_scale), int(prev_y * y_scale)), 
            radius=2, 
            color=(0, 255, 255),  # Yellow
            thickness=-1
        )
    
    # Draw current ball position and info
    ball_results = ball_results_by_frame_id[frame_id]
    for ball_result in ball_results:
        # Get ball coordinates scaled to current frame size
        ball_x = int(ball_result.x_frame * x_scale)
        ball_y = int(ball_result.y_frame * y_scale)
        
        # Add to history
        ball_history.append((ball_result.x_frame, ball_result.y_frame))
        if len(ball_history) > max_history:
            ball_history.pop(0)  # Remove oldest position
        
        # Draw current ball
        cv2.circle(blended, (ball_x, ball_y), radius=10, color=(0, 255, 255), thickness=2)
        
        # Highlight if this is a hit frame
        is_hit = frame_id in hit_frames
        if is_hit:
            cv2.circle(blended, (ball_x, ball_y), radius=20, color=(0, 0, 255), thickness=3)
            cv2.putText(
                blended, "HIT", (ball_x+25, ball_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        
        # Show depth and speed info if available
        if frame_id in ball_info_3d_by_frame:
            draw_ball_info(blended, ball_info_3d_by_frame[frame_id], ball_x, ball_y)
    
    # Add frame counter
    cv2.putText(
        blended, 
        f"Frame: {frame_id}", 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2
    )
    
    return blended

def get_frame_depth_map(
    frame_id: int,
    all_depths: Dict[int, np.ndarray],
    depth_map_cache: Dict[int, np.ndarray],
    ball_info_3d_by_frame: Dict[int, BallInfo3D]
) -> Optional[np.ndarray]:
    """
    Get or compute depth map for a frame (from cache, direct measurement, or interpolation).
    
    Args:
        frame_id: Frame ID.
        all_depths: Dictionary of computed depth maps.
        depth_map_cache: Cache for interpolated depth maps.
        ball_info_3d_by_frame: Dictionary of 3D ball information.
        
    Returns:
        Depth map for the frame, or None if not available.
    """
    # Check cache first for quick retrieval
    if frame_id in depth_map_cache:
        return depth_map_cache[frame_id]
        
    # Use the pre-computed depth map if available
    if frame_id in all_depths:
        depth_map = all_depths[frame_id]
        depth_map_cache[frame_id] = depth_map
        return depth_map
    
    # Check if we have interpolated ball info for this frame - we'll only interpolate depth maps
    # if there's a ball in this frame with an interpolated Z value
    if frame_id in ball_info_3d_by_frame and ball_info_3d_by_frame[frame_id].interpolated:
        # Find the nearest keyframes with depth maps
        all_depth_frames = sorted(all_depths.keys())
        if len(all_depth_frames) >= 2:
            # Find the nearest depth frames before and after
            prev_frames = [f for f in all_depth_frames if f < frame_id]
            next_frames = [f for f in all_depth_frames if f > frame_id]
            
            prev_frame = max(prev_frames) if prev_frames else None
            next_frame = min(next_frames) if next_frames else None
            
            if prev_frame is not None and next_frame is not None:
                # Linear interpolation between depth maps
                prev_depth = all_depths[prev_frame]
                next_depth = all_depths[next_frame]
                
                # Calculate weights based on frame distance
                total_dist = next_frame - prev_frame
                weight_next = (frame_id - prev_frame) / total_dist
                weight_prev = 1 - weight_next
                
                # Interpolate depth map
                depth_map = weight_prev * prev_depth + weight_next * next_depth
                depth_map_cache[frame_id] = depth_map
                return depth_map
            
            # If we only have one direction, use that depth map
            elif prev_frame is not None:
                depth_map = all_depths[prev_frame]
                depth_map_cache[frame_id] = depth_map
                return depth_map
            elif next_frame is not None:
                depth_map = all_depths[next_frame]
                depth_map_cache[frame_id] = depth_map
                return depth_map
    
    return None

def visualize_depth_map(depth_map: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    Create a visualization by blending a depth map with a frame.
    
    Args:
        depth_map: Depth map as numpy array.
        frame: BGR frame to blend with depth map.
        
    Returns:
        Blended visualization frame.
    """
    # Check if depth map needs to be resized - avoid redundant operations
    needs_resize = depth_map.shape[:2] != (frame.shape[0], frame.shape[1])
    
    # Normalize and colorize depth map (once)
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, COLORMAP_DEPTH)
    
    # Resize depth map only if needed
    if needs_resize:
        depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    
    # Use direct blending for better performance
    return cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)

def draw_ball_info(frame: np.ndarray, ball_info: BallInfo3D, x: int, y: int) -> None:
    """
    Draw ball depth and speed information on the frame.
    
    Args:
        frame: Frame to draw on (modified in-place).
        ball_info: Ball information object.
        x: X-coordinate to place text.
        y: Y-coordinate to place text.
    """
    # Mark interpolated depth values differently
    if ball_info.z is not None:
        depth_text = f"Z: {ball_info.z:.2f}" 
        if ball_info.interpolated:
            depth_text += " (interp)"
            text_color = (100, 255, 100)  # Light green for interpolated
        else:
            text_color = (255, 255, 255)  # White for direct measurement
        
        cv2.putText(
            frame, depth_text, (x+15, y+25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
        )
        
        if ball_info.speed_3d is not None:
            speed_text = f"Speed: {ball_info.speed_3d:.2f}"
            cv2.putText(
                frame, speed_text, (x+15, y+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )

def load_model(model_type: str, device: str) -> Any:
    """
    Load the Depth-Anything-V2 model.
    
    Args:
        model_type: Model type ('vits', 'vitb', 'vitl', or 'vitg').
        device: Device to load the model on ('cuda', 'mps', or 'cpu').
        
    Returns:
        Loaded depth model.
        
    Raises:
        ImportError: If the required modules cannot be imported.
        FileNotFoundError: If the model checkpoint file is not found.
        RuntimeError: If the model cannot be loaded.
    """
    # Add the Depth-Anything-V2 repository to Python path
    repo_path = os.path.join(os.getcwd(), "Depth-Anything-V2")
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Depth-Anything-V2 repository not found at {repo_path}")
        
    sys.path.append(repo_path)
    
    # Import the DepthAnythingV2 model
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        logger.info("Successfully imported Depth-Anything-V2 modules")
    except ImportError as e:
        logger.error(f"Error importing Depth-Anything-V2 modules: {e}")
        raise ImportError(f"Failed to import required modules: {e}")
    
    # Validate model type
    if model_type not in MODEL_CONFIGS:
        logger.warning(f"Unknown model type: {model_type}. Using 'vits' instead.")
        model_type = 'vits'
    
    logger.info(f"Using model: {model_type}")
    
    # Check for the checkpoint
    checkpoint_path = f'./checkpoints/depth_anything_v2_{model_type}.pth'
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        download_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_type.replace('vit', '').upper()}/resolve/main/depth_anything_v2_{model_type}.pth?download=true"
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            f"Please download from: {download_url}"
        )
    
    logger.info(f"Found checkpoint at {checkpoint_path}")
    
    try:
        # Initialize the depth model
        logger.info("Initializing model...")
        model = DepthAnythingV2(**MODEL_CONFIGS[model_type])
        
        logger.info("Loading model weights...")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        model = model.to(device).eval()
        
        logger.info("Model successfully loaded")
        return model
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def load_ball_data(ball_json: str, camera_id: str) -> Dict[int, List[DetectionResult]]:
    """
    Load ball detection data from JSON file.
    
    Args:
        ball_json: Path to the ball detection JSON file.
        camera_id: Camera ID to filter detections.
        
    Returns:
        Dictionary mapping frame IDs to ball detection results.
        
    Raises:
        FileNotFoundError: If the JSON file is not found.
        ValueError: If the JSON file is invalid.
    """
    if not os.path.exists(ball_json):
        raise FileNotFoundError(f"Ball detection JSON file not found: {ball_json}")
    
    try:
        logger.info(f"Loading ball detections from {ball_json}...")
        with open(ball_json) as f:
            json_data = json.load(f)
        
        ball_results_by_frame_id = defaultdict(list)
        for dets_per_frame in json_data:
            for cur_det_per_frame in dets_per_frame:
                # Filter for the specified camera and ball detections
                is_valid_detection = (
                    cur_det_per_frame and 
                    cur_det_per_frame.get('camera_id') == camera_id and 
                    cur_det_per_frame.get('detection_class') == 'm_ball'
                )
                
                if is_valid_detection:
                    try:
                        # Create detection object
                        obj = DetectionResult(**cur_det_per_frame)
                        frame_id = obj.frame_id
                        
                        if frame_id is not None:
                            ball_results_by_frame_id[frame_id].append(obj)
                    except Exception as e:
                        logger.warning(f"Skipping invalid detection: {e}")
        
        if not ball_results_by_frame_id:
            logger.warning("No valid ball detections found for the specified camera ID")
        else:
            logger.info(f"Loaded ball detections for {len(ball_results_by_frame_id)} frames")
            
        return ball_results_by_frame_id
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise ValueError(f"Invalid JSON format in {ball_json}: {e}")
    except Exception as e:
        logger.exception(f"Error loading ball detections: {e}")
        raise RuntimeError(f"Failed to load ball detections: {e}")

def main() -> None:
    """
    Main function to run the tennis ball 3D depth analysis.
    
    Handles command-line arguments, loads data, runs inference, and saves results.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tennis Ball 3D Depth Analysis')
    parser.add_argument('--video_path', type=str, default='data/videos/025958_6m47s_7m13s.mp4',
                        help='Path to the video file')
    parser.add_argument('--ball_json', type=str, 
                        default='data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json',
                        help='Path to the ball detection JSON file')
    parser.add_argument('--camera_id', type=str, default='camera1',
                        help='Camera ID in the ball detection data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--model_type', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model type to use')
    parser.add_argument('--max_res', type=int, default=DEFAULT_MAX_RESOLUTION,
                        help='Maximum resolution for processing')
    parser.add_argument('--input_size', type=int, default=DEFAULT_INPUT_SIZE,
                        help='Input size for depth model')
    parser.add_argument('--skip_frames', type=int, default=DEFAULT_SKIP_FRAMES,
                        help='Number of frames to skip (0 = no skipping)')
    parser.add_argument('--max_len', type=int, default=-1,
                        help='Maximum number of frames to process (-1 = all)')
    parser.add_argument('--target_fps', type=float, default=-1,
                        help='Target FPS for output video (-1 = original)')
    parser.add_argument('--no_interpolation', action='store_true',
                        help='Disable depth interpolation')
    parser.add_argument('--ball_depth_radius', type=int, default=DEFAULT_BALL_DEPTH_RADIUS,
                        help='Radius (in pixels) for circular area to calculate mean ball depth')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Validate input files
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found at {args.video_path}")
        if not os.path.exists(args.ball_json):
            raise FileNotFoundError(f"Ball JSON file not found at {args.ball_json}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Set PyTorch to inference mode for better performance
        torch.set_grad_enabled(False)
        
        # Clear CUDA cache before loading model
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Load the model
        model = load_model(args.model_type, device)
        
        # Load ball data
        ball_results_by_frame_id = load_ball_data(args.ball_json, args.camera_id)
        
        # Early filter - detect if we have too many ball detections (might be a misformatted file)
        frame_count = len(ball_results_by_frame_id)
        if frame_count > 10000:
            logger.warning(f"Unusually high number of ball detections ({frame_count} frames). This might slow down processing.")
            
            # Ask user to confirm or exit
            if args.log_level != 'ERROR':
                print(f"The ball detection file contains data for {frame_count} frames, which might cause slow performance.")
                print("Consider filtering the data or processing a shorter segment of the video.")
        
        # Read original video dimensions
        orig_dimensions = get_video_dimensions(args.video_path)
        orig_width, orig_height, orig_fps, orig_total_frames = orig_dimensions
        
        logger.info(f"Original video dimensions: {orig_width}x{orig_height}, FPS: {orig_fps}, Total frames: {orig_total_frames}")
        
        # Processing parameters
        processing_params = {
            'input_size': args.input_size,
            'max_res': args.max_res,
            'max_len': args.max_len,
            'target_fps': args.target_fps,
            'skip_frames': args.skip_frames if not args.no_interpolation else 0,
            'ball_depth_radius': args.ball_depth_radius,
        }
        
        # Set up output paths
        _, video_ext = os.path.splitext(args.video_path)
        base_filename = os.path.basename(args.video_path).replace(video_ext, '')
        
        output_video_avi = os.path.join(args.output_dir, f"{base_filename}_3d_hits.avi")
        output_json_path = os.path.join(args.output_dir, f"{base_filename}_3d_data.json")
        
        # PHASE 1: Depth inference and hit detection with frame skipping
        inference_results = run_depth_inference(
            args.video_path, 
            ball_results_by_frame_id, 
            (orig_width, orig_height),
            model,
            processing_params
        )
        
        # Clear CUDA cache after inference to free memory for visualization
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # PHASE 2: Video visualization
        output_path = create_visualization_video(
            args.video_path,
            inference_results,
            ball_results_by_frame_id,
            (orig_width, orig_height),
            processing_params,
            output_video_avi
        )
        
        # PHASE 3: Save 3D ball data
        save_ball_data(inference_results, output_json_path)
        
        # Final summary
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Processed {inference_results['total_frames']} frames")
        logger.info(f"Detected {len(inference_results['hit_frames'])} ball hits")
        logger.info(f"Output files:")
        logger.info(f"- AVI video: {output_video_avi}")
        logger.info(f"- Ball data: {output_json_path}")
        
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        sys.exit(1)

def get_video_dimensions(video_path: str) -> Tuple[int, int, float, int]:
    """
    Get video dimensions and properties.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Tuple containing (width, height, fps, total_frames).
        
    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, total_frames

def save_ball_data(inference_results: Dict[str, Any], output_json_path: str) -> None:
    """
    Save 3D ball data to a JSON file.
    
    Args:
        inference_results: Results from depth inference.
        output_json_path: Path to save the JSON file.
    """
    logger.info("=" * 50)
    logger.info("PHASE 3: SAVING BALL DATA")
    logger.info("=" * 50)
    
    logger.info("Saving 3D ball data...")
    ball_data = []
    
    ball_info_3d_by_frame = inference_results['ball_info_3d_by_frame']
    hit_frames = inference_results['hit_frames']
    
    for frame_id, ball_info in ball_info_3d_by_frame.items():
        is_hit = frame_id in hit_frames
        ball_data.append({
            "frame_id": frame_id,
            "x": float(ball_info.x),
            "y": float(ball_info.y),
            "z": float(ball_info.z) if ball_info.z is not None else None,
            "timestamp": ball_info.timestamp,
            "speed_3d": ball_info.speed_3d,
            "is_hit": is_hit,
            "interpolated": ball_info.interpolated
        })

    with open(output_json_path, 'w') as f:
        json.dump(ball_data, f, indent=2)

    logger.info(f"3D ball data with hit detection saved to {output_json_path}")

if __name__ == "__main__":
    main()
