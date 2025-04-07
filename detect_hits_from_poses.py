import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, Set, Generator
import numpy as np
import cv2
import argparse
from tqdm.auto import tqdm
import math
from pathlib import Path
import gc
from collections import defaultdict

# Import RTMO model
from rtmo_model import RTMO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_HIT_DISTANCE_THRESHOLD = 120  # Pixels
DEFAULT_MIN_FRAMES_BETWEEN_HITS = 15
DEFAULT_WRIST_CONFIDENCE_THRESHOLD = 0.05
DEFAULT_BALL_HISTORY_LENGTH = 30
DEFAULT_OUTPUT_DIR = "outputs"

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    
    Args:
        obj: Any object potentially containing NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class BallInfo:
    """
    Class to store ball position and trajectory information
    
    Attributes:
        x: X-coordinate of the ball in the frame
        y: Y-coordinate of the ball in the frame
        z: Z-coordinate of the ball (depth) if available
        timestamp: Time when the ball was detected
        speed: Ball speed if available
        is_hit: Flag indicating if this position corresponds to a hit
        interpolated: Flag indicating if this position was interpolated
    """
    def __init__(
        self, 
        x: float, 
        y: float, 
        z: Optional[float] = None, 
        timestamp: Optional[float] = None, 
        speed: Optional[float] = None, 
        is_hit: bool = False, 
        interpolated: bool = False
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z) if z is not None else None
        self.timestamp = timestamp
        self.speed = speed
        self.is_hit = is_hit
        self.interpolated = interpolated
    
    def __repr__(self) -> str:
        """String representation of the ball information"""
        return f"BallInfo(x={self.x:.2f}, y={self.y:.2f}, ts={self.timestamp}, hit={self.is_hit})"

def distance_2d(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate 2D Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def load_ball_data(ball_json: str, camera_id: str = "camera1") -> Optional[Dict[int, BallInfo]]:
    """
    Load ball detection data from original JSON file format
    
    Args:
        ball_json: Path to the ball detection JSON file
        camera_id: Camera ID in the ball detection data
        
    Returns:
        Dictionary of ball information by frame or None if loading failed
    """
    logger.info(f"Loading ball detections from {ball_json}")
    
    if not os.path.exists(ball_json):
        logger.error(f"Ball detection file not found: {ball_json}")
        return None
        
    try:
        with open(ball_json, 'r') as f:
            json_data = json.load(f)
    
        if not json_data:
            logger.warning("Ball detection JSON file is empty")
            return {}
    
        ball_info_by_frame = {}
        total_frames_with_ball = 0
        
        for dets_per_frame in json_data:
            for cur_det_per_frame in dets_per_frame:
                if not cur_det_per_frame:
                    continue
                    
                if 'camera_id' not in cur_det_per_frame or 'detection_class' not in cur_det_per_frame:
                    continue
                    
                if cur_det_per_frame['camera_id'] == camera_id and cur_det_per_frame['detection_class'] == 'm_ball':
                    # Get frame ID
                    if 'frame_id' not in cur_det_per_frame:
                        logger.warning("Found ball detection without frame_id, skipping")
                        continue
                        
                    frame_id = cur_det_per_frame['frame_id']
                    
                    # Check if we have position data
                    if 'x_frame' not in cur_det_per_frame or 'y_frame' not in cur_det_per_frame:
                        logger.warning(f"Missing position data for ball in frame {frame_id}, skipping")
                        continue
                        
                    # Create ball info object
                    ball_info = BallInfo(
                        x=cur_det_per_frame['x_frame'],
                        y=cur_det_per_frame['y_frame'],
                        timestamp=cur_det_per_frame.get('timestamp')
                    )
                    ball_info_by_frame[frame_id] = ball_info
                    total_frames_with_ball += 1
        
        if total_frames_with_ball == 0:
            logger.warning(f"No ball detections found for camera {camera_id}")
            return {}
            
        logger.info(f"Loaded ball detections for {total_frames_with_ball} frames")
        return ball_info_by_frame
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing ball detection JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading ball detections: {e}")
        logger.debug("Stack trace:", exc_info=True)
        return None

def read_video_frames_generator(video_path: str) -> Optional[Generator[Tuple[np.ndarray, int, int, float], None, None]]:
    """
    Generator that yields frames from a video file one at a time to save memory
    
    Args:
        video_path: Path to the video file
        
    Yields:
        Tuple containing:
            - frame: RGB image as numpy array
            - frame_idx: Current frame index
            - frame_count: Total number of frames in the video
            - fps: Frames per second of the video
            
    Returns:
        Generator object or None if video could not be opened
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
        
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
        
        if frame_count <= 0:
            logger.warning("Video has no frames or frame count could not be determined")
            
        # Read frames one at a time
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame at index {frame_idx}, skipping")
                frame_idx += 1
                continue
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            yield frame_rgb, frame_idx, frame_count, fps
            frame_idx += 1
            
        cap.release()
        logger.debug(f"Video frame generator finished after reading {frame_idx} frames")
        
    except Exception as e:
        logger.error(f"Error in read_video_frames_generator: {e}")
        logger.debug("Stack trace:", exc_info=True)
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None

def detect_hits_from_pose(
    video_path: str,
    ball_info_by_frame: Dict[int, BallInfo],
    rtmo_model: RTMO,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect ball hits by analyzing player poses and ball proximity
    
    Args:
        video_path: Path to the video file
        ball_info_by_frame: Dictionary of ball information by frame
        rtmo_model: Initialized RTMO model for pose detection
        params: Dictionary of parameters including:
                - max_frames: Maximum number of frames to process (-1 = all)
                - hit_distance_threshold: Maximum distance in pixels to consider a hit
                - min_frames_between_hits: Minimum number of frames between consecutive hits
                - wrist_confidence_threshold: Minimum confidence for wrist keypoints
        
    Returns:
        Dictionary with detected hits and visualization data
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If frame generation fails
    """
    logger.info("="*50)
    logger.info("PHASE 1: POSE DETECTION AND HIT ANALYSIS")
    logger.info("="*50)
    
    # Validate parameters
    if not ball_info_by_frame:
        raise ValueError("No ball tracking data provided")
        
    # Extract parameters
    max_frames = params.get('max_frames', -1)
    hit_distance_threshold = params.get('hit_distance_threshold', DEFAULT_HIT_DISTANCE_THRESHOLD)
    min_frames_between_hits = params.get('min_frames_between_hits', DEFAULT_MIN_FRAMES_BETWEEN_HITS)
    wrist_confidence_threshold = params.get('wrist_confidence_threshold', DEFAULT_WRIST_CONFIDENCE_THRESHOLD)
    
    # Log parameters
    logger.info(f"Hit detection parameters:")
    logger.info(f"  - Maximum frames to process: {max_frames if max_frames > 0 else 'All'}")
    logger.info(f"  - Hit distance threshold: {hit_distance_threshold} pixels")
    logger.info(f"  - Minimum frames between hits: {min_frames_between_hits}")
    logger.info(f"  - Wrist confidence threshold: {wrist_confidence_threshold}")
    
    # Initialize data structures
    pose_hits = []  # Store frames where hits are detected based on pose
    pose_info_by_frame = {}  # Store pose information by frame
    last_hit_frame = -min_frames_between_hits  # Prevent detecting hits in consecutive frames
    
    # Initialize frame generator
    frame_generator = read_video_frames_generator(video_path)
    if frame_generator is None:
        raise RuntimeError(f"Failed to initialize frame generator for {video_path}")
    
    # Process frames with progress bar
    logger.info(f"Processing {len(ball_info_by_frame)} frames with ball detections")
    progress_bar = tqdm(total=len(ball_info_by_frame), desc="Pose detection and hit analysis")
    
    frames_processed = 0
    frames_with_hits = 0
    
    # Process frames
    try:
        for frame, frame_id, frame_count, fps in frame_generator:
            frames_processed += 1
            
            # Update progress
            progress_bar.update(1)
            
            # Check if we should stop processing
            if max_frames > 0 and frame_id >= max_frames:
                logger.info(f"Reached maximum frames limit ({max_frames})")
                break
                
            # Skip frames without ball data
            if frame_id not in ball_info_by_frame:
                continue
                
            # Get ball information for current frame
            ball_info = ball_info_by_frame[frame_id]
            
            # Run pose detection
            keypoints, scores = rtmo_model(frame)
            
            # Get wrist positions
            wrist_positions, wrist_scores = rtmo_model.get_wrist_positions(
                keypoints, scores, min_score=wrist_confidence_threshold
            )
            
            # Store pose information
            pose_info_by_frame[frame_id] = {
                'keypoints': keypoints,
                'scores': scores,
                'wrists': wrist_positions,
                'wrist_scores': wrist_scores
            }
            
            # Skip hit detection if no wrists detected
            if not wrist_positions:
                logger.debug(f"Frame {frame_id}: No valid wrists detected")
                continue
            
            # Check for hits based on wrist-ball proximity
            hit_detected = False
            closest_wrist_distance = float('inf')
            closest_wrist_idx = -1
            
            for wrist_idx, wrist_pos in enumerate(wrist_positions):
                # Get ball position (2D only)
                ball_pos = [ball_info.x, ball_info.y]
                
                # Calculate 2D distance
                distance = distance_2d(wrist_pos, ball_pos)
                
                if distance < closest_wrist_distance:
                    closest_wrist_distance = distance
                    closest_wrist_idx = wrist_idx
                    
            # Detect hit if the closest wrist is within the threshold distance
            # and enough frames have passed since the last detected hit
            if (closest_wrist_idx >= 0 and 
                closest_wrist_distance < hit_distance_threshold and 
                frame_id - last_hit_frame >= min_frames_between_hits):
                hit_detected = True
                frames_with_hits += 1
                last_hit_frame = frame_id
                
                # Add to hits list
                hit_info = {
                    'frame_id': frame_id,
                    'timestamp': ball_info.timestamp,
                    'ball_position': [ball_info.x, ball_info.y],
                    'wrist_position': wrist_positions[closest_wrist_idx].tolist(),
                    'distance': float(closest_wrist_distance),
                    'wrist_score': float(wrist_scores[closest_wrist_idx])
                }
                pose_hits.append(hit_info)
                
                logger.info(f"Hit detected at frame {frame_id} (distance: {closest_wrist_distance:.2f} px)")
            
            # Free memory periodically
            if frame_id % 20 == 0:
                gc.collect()
    
    except Exception as e:
        logger.error(f"Error during pose detection: {e}")
        logger.debug("Stack trace:", exc_info=True)
        # Close progress bar before raising exception
        progress_bar.close()
        raise RuntimeError(f"Hit detection failed: {e}")
    
    # Close progress bar
    progress_bar.close()
    
    # Log hit detection results
    if frames_with_hits > 0:
        logger.info(f"Detected {frames_with_hits} hits based on player pose and ball proximity")
        logger.info(f"Hit frames: {[hit['frame_id'] for hit in pose_hits]}")
    else:
        logger.warning("No hits detected in the video")
    
    logger.info(f"Processed {frames_processed} frames, {len(pose_info_by_frame)} with pose data")
    
    return {
        'pose_hits': pose_hits,
        'pose_info_by_frame': pose_info_by_frame
    }

def create_visualization_video(
    video_path: str, 
    ball_info_by_frame: Dict[int, BallInfo], 
    pose_results: Dict[str, Any], 
    orig_dimensions: Tuple[int, int], 
    params: Dict[str, Any], 
    output_path: str
) -> str:
    """
    Create a visualization video showing ball tracking and player poses with hit detection
    
    Args:
        video_path: Path to the video file
        ball_info_by_frame: Dictionary of ball information by frame
        pose_results: Results from pose detection
        orig_dimensions: Original video dimensions (width, height)
        params: Dictionary of parameters including:
                - max_frames: Maximum number of frames to process (-1 = all)
        output_path: Path to save the output video
        
    Returns:
        Path to the output video
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If video processing fails
    """
    logger.info("="*50)
    logger.info("PHASE 2: VIDEO VISUALIZATION")
    logger.info("="*50)
    
    # Validate inputs
    if not ball_info_by_frame:
        raise ValueError("No ball tracking data provided")
        
    if 'pose_hits' not in pose_results or 'pose_info_by_frame' not in pose_results:
        raise ValueError("Invalid pose detection results")
    
    # Extract results
    pose_hits = pose_results['pose_hits']
    pose_info_by_frame = pose_results['pose_info_by_frame']
    
    # Extract parameters
    max_frames = params.get('max_frames', -1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Initialize frame generator
    logger.info(f"Creating visualization video from {video_path}")
    frame_generator = read_video_frames_generator(video_path)
    if frame_generator is None:
        raise RuntimeError(f"Failed to initialize frame generator for {video_path}")
    
    # Get first frame to set up dimensions
    try:
        frame_info = next(frame_generator, None)
        if frame_info is None:
            raise RuntimeError("Failed to read first frame from video")
        
        frame, frame_id, frame_count, fps = frame_info
        height, width = frame.shape[:2]
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
        
        logger.info(f"Creating visualization video with dimensions {width}x{height} @ {fps:.2f}fps")
        
        # Reset ball history for visualization
        ball_history = []
        max_history = DEFAULT_BALL_HISTORY_LENGTH
        
        # Initialize progress bar
        progress_bar = tqdm(total=frame_count, desc="Video visualization")
        
        # Find hit frames for quick lookup
        hit_frames = set(hit['frame_id'] for hit in pose_hits)
        logger.info(f"Visualizing {len(hit_frames)} hit frames")
        
        # Process first frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        visualize_frame(frame_bgr, frame_id, ball_info_by_frame, pose_info_by_frame, hit_frames, ball_history)
        video_writer.write(frame_bgr)
        progress_bar.update(1)
        frames_processed = 1
        
        # Process the rest of the frames
        for frame, frame_id, _, _ in frame_generator:
            frames_processed += 1
            
            # Check if we should stop processing
            if max_frames > 0 and frame_id >= max_frames:
                logger.info(f"Reached maximum frames limit ({max_frames})")
                break
                
            # Convert the frame to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw visualizations on the frame
            visualize_frame(frame_bgr, frame_id, ball_info_by_frame, pose_info_by_frame, hit_frames, ball_history)
            
            # Write the frame to video
            video_writer.write(frame_bgr)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Free memory periodically
            if frame_id % 20 == 0:
                gc.collect()
        
        # Close progress bar
        progress_bar.close()
        
        # Release the video writer
        video_writer.release()
        logger.info(f"Visualization video saved to {output_path}")
        logger.info(f"Processed {frames_processed} frames")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating visualization video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        
        # Clean up resources
        if 'progress_bar' in locals():
            progress_bar.close()
            
        if 'video_writer' in locals() and video_writer.isOpened():
            video_writer.release()
            
        raise RuntimeError(f"Video visualization failed: {e}")

def visualize_frame(
    frame: np.ndarray, 
    frame_id: int, 
    ball_info_by_frame: Dict[int, BallInfo], 
    pose_info_by_frame: Dict[int, Dict[str, Any]], 
    hit_frames: Set[int], 
    ball_history: List[Tuple[float, float]]
) -> None:
    """
    Draw visualizations on a video frame
    
    Args:
        frame: The frame to draw on (BGR format)
        frame_id: Current frame ID
        ball_info_by_frame: Dictionary of ball information by frame
        pose_info_by_frame: Dictionary of pose information by frame
        hit_frames: Set of frame IDs where hits were detected
        ball_history: List to track ball position history
    """
    height, width = frame.shape[:2]
    
    # Draw pose keypoints if available
    if frame_id in pose_info_by_frame:
        pose_info = pose_info_by_frame[frame_id]
        keypoints = pose_info['keypoints']
        scores = pose_info['scores']
        wrists = pose_info.get('wrists', [])
        
        # Define the connections between keypoints to form a skeleton
        skeleton_connections = [
            # Face connections
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Body
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        # Draw skeleton lines for each person
        for person_idx in range(keypoints.shape[0]):
            # Draw skeleton lines
            for start_idx, end_idx in skeleton_connections:
                if scores[person_idx, start_idx] > 0.3 and scores[person_idx, end_idx] > 0.3:
                    try:
                        start_pt = tuple(map(int, keypoints[person_idx, start_idx]))
                        end_pt = tuple(map(int, keypoints[person_idx, end_idx]))
                        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
                    except (ValueError, IndexError, TypeError) as e:
                        # Skip drawing this line but continue with the rest
                        logger.debug(f"Error drawing skeleton line: {e}")
                        continue
            
            # Draw keypoints
            for kp_idx in range(keypoints.shape[1]):
                try:
                    x, y = keypoints[person_idx, kp_idx]
                    conf = scores[person_idx, kp_idx]
                    
                    if conf > 0.3:
                        # Draw all keypoints in green, but wrists in different color
                        color = (0, 0, 255) if kp_idx in [9, 10] else (0, 255, 0)
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                except (ValueError, IndexError, TypeError) as e:
                    # Skip drawing this keypoint but continue with the rest
                    logger.debug(f"Error drawing keypoint: {e}")
                    continue
    
    # Draw ball position and history
    if frame_id in ball_info_by_frame:
        ball_info = ball_info_by_frame[frame_id]
        
        try:
            # Add to history
            if hasattr(ball_info, 'x') and hasattr(ball_info, 'y'):
                ball_history.append((ball_info.x, ball_info.y))
                if len(ball_history) > DEFAULT_BALL_HISTORY_LENGTH:
                    ball_history.pop(0)  # Remove oldest position
                    
                # Draw trajectory
                for prev_x, prev_y in ball_history:
                    cv2.circle(frame, (int(prev_x), int(prev_y)), radius=2, color=(0, 255, 255), thickness=-1)
                    
                # Draw current ball
                cv2.circle(frame, (int(ball_info.x), int(ball_info.y)), radius=10, color=(0, 255, 255), thickness=2)
                
                # Highlight hit frames
                if frame_id in hit_frames:
                    # Draw a larger circle for hits
                    cv2.circle(frame, (int(ball_info.x), int(ball_info.y)), radius=20, color=(0, 0, 255), thickness=3)
                    
                    # Add "HIT" text
                    cv2.putText(frame, "HIT", (int(ball_info.x)+25, int(ball_info.y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except (ValueError, TypeError) as e:
            logger.debug(f"Error drawing ball: {e}")
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main() -> int:
    """
    Main function to run the tennis hit detection pipeline
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(
            description='Tennis Hit Detection Using Player Pose and Ball Tracking',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--video_path', type=str, required=True,
                            help='Path to the video file')
        parser.add_argument('--ball_json', type=str, required=True,
                            help='Path to the original ball detection JSON file')
        parser.add_argument('--camera_id', type=str, default='camera1',
                            help='Camera ID in the ball detection data')
        parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                            help='Directory to save outputs')
        parser.add_argument('--rtmo_model', type=str, default=None,
                            help='Path to RTMO model file (default is rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx in current dir)')
        parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                            help='Device to run inference on (cpu or cuda)')
        parser.add_argument('--hit_distance', type=int, default=DEFAULT_HIT_DISTANCE_THRESHOLD,
                            help='Maximum distance (in pixels) between wrist and ball to consider a hit')
        parser.add_argument('--min_frames_between_hits', type=int, default=DEFAULT_MIN_FRAMES_BETWEEN_HITS,
                            help='Minimum number of frames between consecutive hits')
        parser.add_argument('--max_frames', type=int, default=-1,
                            help='Maximum number of frames to process (-1 = all)')
        parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                            help='Set the logging level')
        
        # Parse arguments
        args = parser.parse_args()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Running hit detection with log level: {args.log_level}")
        
        # Verify input files exist
        if not os.path.exists(args.video_path):
            logger.error(f"Video file not found at {args.video_path}")
            return 1
            
        if not os.path.exists(args.ball_json):
            logger.error(f"Ball detection JSON file not found at {args.ball_json}")
            return 1
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        
        # Load ball tracking data
        logger.info("Loading ball tracking data...")
        ball_info_by_frame = load_ball_data(args.ball_json, args.camera_id)
        if ball_info_by_frame is None:
            logger.error("Failed to load ball data")
            return 1
            
        if not ball_info_by_frame:
            logger.warning("No ball data found in the input file")
            return 1
        
        # Initialize RTMO model
        logger.info("Initializing RTMO model...")
        try:
            rtmo_model = RTMO(model_path=args.rtmo_model, device=args.device)
            logger.info("RTMO model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RTMO model: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return 1
        
        # Read original video dimensions
        try:
            cap = cv2.VideoCapture(args.video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {args.video_path}")
                return 1
                
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            logger.info(f"Original video dimensions: {orig_width}x{orig_height}, Total frames: {orig_total_frames}")
        except Exception as e:
            logger.error(f"Error reading video properties: {e}")
            return 1
        
        # Processing parameters
        processing_params = {
            'max_frames': args.max_frames,
            'hit_distance_threshold': args.hit_distance,
            'min_frames_between_hits': args.min_frames_between_hits,
            'wrist_confidence_threshold': DEFAULT_WRIST_CONFIDENCE_THRESHOLD,
        }
        
        # Set up output paths
        _, video_ext = os.path.splitext(args.video_path)
        base_filename = os.path.basename(args.video_path).replace(video_ext, '')
        
        output_video_avi = os.path.join(args.output_dir, f"{base_filename}_pose_hits.avi")
        output_json_path = os.path.join(args.output_dir, f"{base_filename}_pose_hits.json")
        
        logger.info(f"Processing video: {args.video_path}")
        logger.info(f"Output paths:")
        logger.info(f"  - AVI video: {output_video_avi}")
        logger.info(f"  - JSON data: {output_json_path}")
        
        try:
            # PHASE 1: Pose detection and hit analysis
            pose_results = detect_hits_from_pose(
                args.video_path,
                ball_info_by_frame,
                rtmo_model,
                processing_params
            )
            
            # PHASE 2: Video visualization
            output_path = create_visualization_video(
                args.video_path,
                ball_info_by_frame,
                pose_results,
                (orig_width, orig_height),
                processing_params,
                output_video_avi
            )
            
            # PHASE 3: Save hit data to JSON
            logger.info("="*50)
            logger.info("PHASE 3: SAVING HIT DATA")
            logger.info("="*50)
            
            if not pose_results['pose_hits']:
                logger.warning("No hits were detected, saving empty hit data")
                
            # Convert NumPy values to Python native types for JSON serialization
            serializable_hits = convert_numpy_types(pose_results['pose_hits'])
            
            # Add metadata to the output
            output_data = {
                'metadata': {
                    'video_path': os.path.abspath(args.video_path),
                    'ball_json': os.path.abspath(args.ball_json),
                    'camera_id': args.camera_id,
                    'hit_distance_threshold': args.hit_distance,
                    'min_frames_between_hits': args.min_frames_between_hits,
                    'total_frames_processed': len(pose_results['pose_info_by_frame']),
                    'timestamp': str(Path(output_json_path).stat().st_mtime) if Path(output_json_path).exists() else None
                },
                'hits': serializable_hits
            }
            
            # Save to JSON file
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Hit data saved to {output_json_path}")
            
            # Final summary
            logger.info("="*50)
            logger.info("PROCESSING COMPLETE")
            logger.info("="*50)
            logger.info(f"Detected {len(pose_results['pose_hits'])} hits based on player pose and ball proximity")
            logger.info("Output files:")
            logger.info(f"- AVI video: {output_video_avi}")
            logger.info(f"- Hit data: {output_json_path}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return 1
            
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        logger.debug("Stack trace:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
