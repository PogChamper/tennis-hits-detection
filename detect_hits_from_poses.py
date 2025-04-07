import os
import sys
import json
import numpy as np
import cv2
import argparse
from tqdm.auto import tqdm
import math
from pathlib import Path
import subprocess
import gc
from collections import defaultdict

# Import RTMO model
from rtmo_model import RTMO

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
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
    """Class to store ball position and trajectory information"""
    def __init__(self, x, y, z=None, timestamp=None, speed=None, is_hit=False, interpolated=False):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z) if z is not None else None
        self.timestamp = timestamp
        self.speed = speed
        self.is_hit = is_hit
        self.interpolated = interpolated

def distance_2d(point1, point2):
    """Calculate 2D Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def load_ball_data(ball_json, camera_id="camera1"):
    """
    Load ball detection data from original JSON file format
    
    Args:
        ball_json: Path to the ball detection JSON file
        camera_id: Camera ID in the ball detection data
        
    Returns:
        Dictionary of ball information by frame
    """
    print(f"Loading ball detections from {ball_json}...")
    try:
        with open(ball_json) as f:
            json_data = json.load(f)
    
        ball_info_by_frame = {}
        for dets_per_frame in json_data:
            for cur_det_per_frame in dets_per_frame:
                if cur_det_per_frame and 'camera_id' in cur_det_per_frame and 'detection_class' in cur_det_per_frame:
                    if cur_det_per_frame['camera_id'] == camera_id and cur_det_per_frame['detection_class'] == 'm_ball':
                        frame_id = cur_det_per_frame['frame_id']
                        
                        # Create ball info object
                        ball_info = BallInfo(
                            x=cur_det_per_frame['x_frame'],
                            y=cur_det_per_frame['y_frame'],
                            timestamp=cur_det_per_frame.get('timestamp')
                        )
                        ball_info_by_frame[frame_id] = ball_info
        
        print(f"Loaded ball detections for {len(ball_info_by_frame)} frames")
        return ball_info_by_frame
    
    except Exception as e:
        print(f"Error loading ball detections: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_video_frames_generator(video_path):
    """Generator that yields frames from a video file one at a time to save memory"""
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")
        
        # Read frames one at a time
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            yield frame_rgb, frame_idx, frame_count, fps
            frame_idx += 1
            
        cap.release()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in read_video_frames_generator: {e}")
        return None

def detect_hits_from_pose(video_path, ball_info_by_frame, rtmo_model, params):
    """
    Detect ball hits by analyzing player poses and ball proximity
    
    Args:
        video_path: Path to the video file
        ball_info_by_frame: Dictionary of ball information by frame
        rtmo_model: Initialized RTMO model for pose detection
        params: Dictionary of parameters
        
    Returns:
        Dictionary with detected hits and visualization data
    """
    print("\n" + "="*50)
    print("PHASE 1: POSE DETECTION AND HIT ANALYSIS")
    print("="*50)
    
    # Extract parameters
    max_frames = params.get('max_frames', -1)
    hit_distance_threshold = params.get('hit_distance_threshold', 60)  # Pixels
    min_frames_between_hits = params.get('min_frames_between_hits', 15)
    wrist_confidence_threshold = params.get('wrist_confidence_threshold', 0.05)
    
    # Initialize data structures
    pose_hits = []  # Store frames where hits are detected based on pose
    pose_info_by_frame = {}  # Store pose information by frame
    last_hit_frame = -min_frames_between_hits  # Prevent detecting hits in consecutive frames
    
    # Initialize frame generator
    frame_generator = read_video_frames_generator(video_path)
    if frame_generator is None:
        raise Exception("Failed to initialize frame generator")
    
    # Process frames with progress bar
    progress_bar = tqdm(total=len(ball_info_by_frame), desc="Pose detection and hit analysis")
    
    # Process frames
    for frame, frame_id, frame_count, fps in frame_generator:
        # Update progress
        progress_bar.update(1)
        
        # Check if we should stop processing
        if max_frames > 0 and frame_id >= max_frames:
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
        if closest_wrist_idx >= 0 and (closest_wrist_distance < hit_distance_threshold and 
            frame_id - last_hit_frame >= min_frames_between_hits):
            hit_detected = True
            last_hit_frame = frame_id
            
            # Add to hits list
            hit_info = {
                'frame_id': frame_id,
                'timestamp': ball_info.timestamp,
                'ball_position': [ball_info.x, ball_info.y],
                'wrist_position': wrist_positions[closest_wrist_idx].tolist(),
                'distance': closest_wrist_distance,
                'wrist_score': wrist_scores[closest_wrist_idx]
            }
            pose_hits.append(hit_info)
            
            print(f"Hit detected at frame {frame_id} (distance: {closest_wrist_distance:.2f} px)")
        
        # Free memory periodically
        if frame_id % 20 == 0:
            gc.collect()
    
    # Close progress bar
    progress_bar.close()
    
    print(f"Detected {len(pose_hits)} hits based on player pose and ball proximity")
    
    return {
        'pose_hits': pose_hits,
        'pose_info_by_frame': pose_info_by_frame
    }

def create_visualization_video(video_path, ball_info_by_frame, pose_results, orig_dimensions, params, output_path):
    """
    Create a visualization video showing ball tracking and player poses with hit detection
    
    Args:
        video_path: Path to the video file
        ball_info_by_frame: Dictionary of ball information by frame
        pose_results: Results from pose detection
        orig_dimensions: Original video dimensions (width, height)
        params: Dictionary of parameters
        output_path: Path to save the output video
        
    Returns:
        Path to the output video
    """
    print("\n" + "="*50)
    print("PHASE 2: VIDEO VISUALIZATION")
    print("="*50)
    
    # Extract results
    pose_hits = pose_results['pose_hits']
    pose_info_by_frame = pose_results['pose_info_by_frame']
    
    # Extract parameters
    max_frames = params.get('max_frames', -1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize frame generator
    frame_generator = read_video_frames_generator(video_path)
    if frame_generator is None:
        raise Exception("Failed to initialize frame generator")
    
    # Get first frame to set up dimensions
    frame_info = next(frame_generator, None)
    if frame_info is None:
        raise Exception("Failed to read first frame")
    
    frame, frame_id, frame_count, fps = frame_info
    height, width = frame.shape[:2]
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise Exception(f"Failed to open VideoWriter for {output_path}")
    
    print(f"Creating visualization video with dimensions {width}x{height} @ {fps}fps")
    
    # Reset ball history for visualization
    ball_history = []
    max_history = 30  # Maximum number of past positions to display
    
    # Initialize progress bar
    progress_bar = tqdm(total=frame_count, desc="Video visualization")
    
    # Find hit frames for quick lookup
    hit_frames = [hit['frame_id'] for hit in pose_hits]
    
    # Process first frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    visualize_frame(frame_bgr, frame_id, ball_info_by_frame, pose_info_by_frame, hit_frames, ball_history)
    video_writer.write(frame_bgr)
    progress_bar.update(1)
    
    # Process the rest of the frames
    for frame, frame_id, _, _ in frame_generator:
        # Check if we should stop processing
        if max_frames > 0 and frame_id >= max_frames:
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
    print(f"Visualization video saved to {output_path}")
    
    return output_path

def visualize_frame(frame, frame_id, ball_info_by_frame, pose_info_by_frame, hit_frames, ball_history):
    """
    Draw visualizations on a video frame
    
    Args:
        frame: The frame to draw on (BGR format)
        frame_id: Current frame ID
        ball_info_by_frame: Dictionary of ball information by frame
        pose_info_by_frame: Dictionary of pose information by frame
        hit_frames: List of frame IDs where hits were detected
        ball_history: List to track ball position history
    """
    height, width = frame.shape[:2]
    
    # Draw pose keypoints if available
    if frame_id in pose_info_by_frame:
        pose_info = pose_info_by_frame[frame_id]
        keypoints = pose_info['keypoints']
        scores = pose_info['scores']
        wrists = pose_info['wrists']
        
        # Draw skeleton lines for each person
        for person_idx in range(keypoints.shape[0]):
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
            
            # Draw skeleton lines
            for start_idx, end_idx in skeleton_connections:
                if scores[person_idx, start_idx] > 0.3 and scores[person_idx, end_idx] > 0.3:
                    start_pt = tuple(map(int, keypoints[person_idx, start_idx]))
                    end_pt = tuple(map(int, keypoints[person_idx, end_idx]))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            
            # Draw keypoints
            for kp_idx in range(keypoints.shape[1]):
                x, y = keypoints[person_idx, kp_idx]
                conf = scores[person_idx, kp_idx]
                
                if conf > 0.3:
                    # Draw all keypoints in green, but wrists in different color
                    color = (0, 0, 255) if kp_idx in [9, 10] else (0, 255, 0)
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    
    # Draw ball position and history
    if frame_id in ball_info_by_frame:
        ball_info = ball_info_by_frame[frame_id]
        
        # Add to history
        ball_history.append((ball_info.x, ball_info.y))
        if len(ball_history) > 30:
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
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def convert_to_mp4(input_path, output_path):
    """Convert video to MP4 format using FFmpeg"""
    print("\n" + "="*50)
    print("PHASE 3: VIDEO CONVERSION")
    print("="*50)
    
    try:
        print("Converting to MP4 with FFmpeg...")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Converted video saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to convert video with FFmpeg: {e}")
        print("The AVI version of the video should still be available.")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tennis Hit Detection Using Player Pose and Ball Tracking')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the video file')
    parser.add_argument('--ball_json', type=str, required=True,
                        help='Path to the original ball detection JSON file')
    parser.add_argument('--camera_id', type=str, default='camera1',
                        help='Camera ID in the ball detection data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--rtmo_model', type=str, default=None,
                        help='Path to RTMO model file (default is rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx in current dir)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--hit_distance', type=int, default=120,
                        help='Maximum distance (in pixels) between wrist and ball to consider a hit')
    parser.add_argument('--min_frames_between_hits', type=int, default=15,
                        help='Minimum number of frames between consecutive hits')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Maximum number of frames to process (-1 = all)')
    parser.add_argument('--no_mp4_convert', action='store_true',
                        help='Disable MP4 conversion')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video file not found at {args.video_path}")
        sys.exit(1)
    if not os.path.exists(args.ball_json):
        print(f"ERROR: Ball detection JSON file not found at {args.ball_json}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ball tracking data
    ball_info_by_frame = load_ball_data(args.ball_json, args.camera_id)
    if ball_info_by_frame is None:
        print("Failed to load ball data")
        sys.exit(1)
    
    # Initialize RTMO model
    print("Initializing RTMO model...")
    try:
        rtmo_model = RTMO(model_path=args.rtmo_model, device=args.device)
        print("RTMO model initialized successfully")
    except Exception as e:
        print(f"Error initializing RTMO model: {e}")
        sys.exit(1)
    
    # Read original video dimensions
    cap = cv2.VideoCapture(args.video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Original video dimensions: {orig_width}x{orig_height}, Total frames: {orig_total_frames}")
    
    # Processing parameters
    processing_params = {
        'max_frames': args.max_frames,
        'hit_distance_threshold': args.hit_distance,
        'min_frames_between_hits': args.min_frames_between_hits,
        'wrist_confidence_threshold': 0.05,
    }
    
    # Set up output paths
    _, video_ext = os.path.splitext(args.video_path)
    base_filename = os.path.basename(args.video_path).replace(video_ext, '')
    
    output_video_avi = os.path.join(args.output_dir, f"{base_filename}_pose_hits.avi")
    output_video_mp4 = os.path.join(args.output_dir, f"{base_filename}_pose_hits.mp4")
    output_json_path = os.path.join(args.output_dir, f"{base_filename}_pose_hits.json")
    
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
        
        # PHASE 3: Convert to MP4 (optional)
        if not args.no_mp4_convert:
            convert_to_mp4(output_video_avi, output_video_mp4)
        
        # PHASE 4: Save hit data to JSON
        print("\n" + "="*50)
        print("PHASE 4: SAVING HIT DATA")
        print("="*50)
        
        # Convert NumPy values to Python native types for JSON serialization
        serializable_hits = convert_numpy_types(pose_results['pose_hits'])
        
        with open(output_json_path, 'w') as f:
            json.dump(serializable_hits, f, indent=2)
        
        print(f"Hit data saved to {output_json_path}")
        
        # Final summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Detected {len(pose_results['pose_hits'])} hits based on player pose and ball proximity")
        print(f"Output files:")
        print(f"- AVI video: {output_video_avi}")
        if os.path.exists(output_video_mp4):
            print(f"- MP4 video: {output_video_mp4}")
        print(f"- Hit data: {output_json_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 