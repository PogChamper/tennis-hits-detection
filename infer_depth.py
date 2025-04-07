import os
import sys
import torch
import gc  # For garbage collection
from collections import defaultdict
import json
import typing as t
import numpy as np
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pydantic import BaseModel
import math
from scipy.interpolate import interp1d  # For interpolation
import argparse

class DetectionResult(BaseModel):
    detection_class: str
    detection_class_id: int
    object_number: int
    track_id_raw: int
    track_id: int
    score: float
    box: t.Optional[t.Tuple[int, int, int, int]] = None
    box_rel: t.Optional[t.Tuple[float, float, float, float]] = None
    x_frame: t.Optional[float] = None
    y_frame: t.Optional[float] = None
    x_court: t.Optional[float] = None
    y_court: t.Optional[float] = None
    z_court: t.Optional[float] = None
    visible_status: t.Optional[int] = None
    meta: t.Optional[t.Dict] = None
    frame_id: t.Optional[int] = None
    camera_id: t.Optional[str] = None
    timestamp: t.Optional[float] = None

# Custom implementation of read_video_frames
def read_video_frames_generator(video_path, process_length=-1, target_fps=-1, max_res=-1, skip_frames=0):
    """
    Generator that yields frames from a video file one at a time to save memory.
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {original_width}x{original_height} @ {original_fps}fps, {frame_count} frames")
        
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
        
        print(f"Target size: {width}x{height}, Target FPS: {fps}, Stride: {stride}, Skip frames: {skip_frames}")
        
        # Calculate number of frames to read
        if process_length <= 0:
            process_length = frame_count
        else:
            process_length = min(process_length, frame_count)
            
        # Number of frames to capture after stride
        num_frames = min(process_length, 1 + (frame_count - 1) // stride)
        print(f"Will process {num_frames} frames...")
        
        # Read frames one at a time
        frame_idx = 0
        yielded_frames = 0
        skipped_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or yielded_frames >= num_frames:
                break
                
            if frame_idx % stride == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if width != original_width or height != original_height:
                    frame = cv2.resize(frame, (width, height))
                
                # Determine if this is a keyframe or should be skipped for depth inference
                is_keyframe = (skipped_count % (skip_frames + 1)) == 0
                
                yield frame, yielded_frames, num_frames, fps, is_keyframe
                yielded_frames += 1
                skipped_count += 1
                
            frame_idx += 1
            
        cap.release()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in read_video_frames_generator: {e}")
        return None

# Class to store ball information with 3D coordinates and speed
class BallInfo3D:
    def __init__(self, x, y, z=None, timestamp=None, interpolated=False):
        self.x = x
        self.y = y
        self.z = z
        self.timestamp = timestamp
        self.speed_3d = None
        self.interpolated = interpolated
        
    def calculate_speed(self, prev_ball):
        if prev_ball and prev_ball.z is not None and self.z is not None and prev_ball.timestamp is not None:
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
def interpolate_z_values(ball_info_dict, all_frame_ids):
    """
    Interpolate Z values for frames between keyframes.
    """
    # Get all frames with ball info
    frame_ids_with_info = sorted(ball_info_dict.keys())
    
    if len(frame_ids_with_info) < 2:
        return ball_info_dict  # Not enough data to interpolate
    
    # First, gather existing z values and timestamps
    existing_frame_ids = []
    existing_z_values = []
    existing_timestamps = []
    
    for frame_id in frame_ids_with_info:
        ball_info = ball_info_dict[frame_id]
        if ball_info.z is not None:
            existing_frame_ids.append(frame_id)
            existing_z_values.append(ball_info.z)
            existing_timestamps.append(ball_info.timestamp if ball_info.timestamp is not None else frame_id)
    
    if len(existing_z_values) < 2:
        return ball_info_dict  # Not enough data with valid Z values to interpolate
    
    # Create interpolation function - linear interpolation
    z_interp = interp1d(existing_timestamps, existing_z_values, kind='cubic', 
                       bounds_error=False, fill_value=(existing_z_values[0], existing_z_values[-1]))
    
    # Create interpolated entries for frames that don't have ball info
    for frame_id in all_frame_ids:
        if frame_id in ball_info_dict and ball_info_dict[frame_id].z is not None:
            # Skip frames that already have valid depth information
            continue
            
        # Find the ball detection for this frame
        ball_detection = None
        for detection_frame_id in frame_ids_with_info:
            if detection_frame_id == frame_id:
                ball_detection = ball_info_dict[detection_frame_id]
                break
                
        if ball_detection is None:
            # No ball detected in this frame, so skip
            continue
            
        # Get the timestamp for this frame (use frame_id if timestamp not available)
        timestamp = ball_detection.timestamp if ball_detection.timestamp is not None else frame_id
        
        # Interpolate Z value
        interpolated_z = float(z_interp(timestamp))
        
        # Create or update ball info for this frame
        if frame_id in ball_info_dict:
            # Update existing entry
            ball_info_dict[frame_id].z = interpolated_z
            ball_info_dict[frame_id].interpolated = True
        else:
            # Create new entry (should not happen in practice)
            ball_info_dict[frame_id] = BallInfo3D(
                ball_detection.x, ball_detection.y, 
                interpolated_z, ball_detection.timestamp, 
                interpolated=True
            )
    
    return ball_info_dict

# Function to detect hits from 3D ball trajectory
def detect_hits(ball_info_3d_by_frame, fps=30, min_angle_change=85, min_speed_change=400):
    """
    Detect moments when players hit the ball by analyzing 3D trajectory and speed changes.
    """
    hits = []
    frame_ids = sorted(ball_info_3d_by_frame.keys())
    
    if len(frame_ids) < 3:
        return hits
    
    # Calculate velocity vectors in 3D
    velocity_vectors = []
    for i in range(1, len(frame_ids)):
        prev_frame_id = frame_ids[i-1]
        curr_frame_id = frame_ids[i]
        
        prev_ball = ball_info_3d_by_frame[prev_frame_id]
        curr_ball = ball_info_3d_by_frame[curr_frame_id]
        
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
    
    # Detect significant direction or speed changes
    window_size = 3  # Use a small window for smoothing
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
            if angle > min_angle_change or (speed_change is not None and speed_change > min_speed_change):
                hits.append((curr_frame_id, angle, speed_change))
    
    return [hit[0] for hit in hits]

# First, let's add a new function to calculate mean depth in a circular area
def get_circular_mean_depth(depth_map, center_x, center_y, radius=5):
    """
    Calculate the mean depth value within a circular area around a point.
    
    Args:
        depth_map: The depth map array
        center_x, center_y: Center coordinates of the circle
        radius: Radius of the circle in pixels
    
    Returns:
        Mean depth value within the circle, or None if no valid depths are found
    """
    height, width = depth_map.shape
    depths = []
    
    # Iterate through a square bounding the circle
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            # Check if the point is within the circle
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                depth_val = depth_map[y, x]
                if depth_val is not None and not np.isnan(depth_val):
                    depths.append(depth_val)
    
    # Return mean if we have valid depths, None otherwise
    return np.mean(depths) if depths else None

# Then modify the run_depth_inference function to use this new function
def run_depth_inference(video_path, ball_results_by_frame_id, orig_dimensions, model, params):
    print("\n" + "="*50)
    print("PHASE 1: DEPTH INFERENCE WITH FRAME SKIPPING")
    print("="*50)
    
    max_len = params.get('max_len', -1)
    target_fps = params.get('target_fps', -1)
    max_res = params.get('max_res', 960)
    input_size = params.get('input_size', 720)
    skip_frames = params.get('skip_frames', 3)  # Default to processing every 4th frame
    ball_depth_radius = params.get('ball_depth_radius', 5)  # Radius for circular mean depth calculation
    
    orig_width, orig_height = orig_dimensions
    
    # Initialize depth storage for ball tracking
    all_depths = {}  # Store only frames with ball detections
    all_frame_data = {}  # Store all frame metadata for interpolation
    ball_info_3d_by_frame = {}
    previous_ball_info = None
    frame_dimensions = None
    video_fps = None
    total_frames_count = None
    
    # First pass: Compute depths and ball tracking data
    frame_generator = read_video_frames_generator(video_path, max_len, target_fps, max_res, skip_frames)
    if frame_generator is None:
        raise Exception("Failed to initialize frame generator")
    
    # Get first frame to set up dimensions
    frame_info = next(frame_generator, None)
    if frame_info is None:
        raise Exception("Failed to read first frame")
    
    first_frame, first_frame_idx, total_frames, fps, is_keyframe = frame_info
    frame_dimensions = first_frame.shape[:2]  # (height, width)
    video_fps = fps
    total_frames_count = total_frames
    
    # Store frame data for all frames to help with interpolation
    all_frame_data[first_frame_idx] = {'is_keyframe': is_keyframe}
    
    # Process all frames with progress bar
    print(f"Running depth inference on {total_frames} frames (processing 1 of every {skip_frames+1} frames)...")
    
    # Initialize progress bar for inference
    progress_bar = tqdm(total=total_frames, desc="Depth inference")
    
    # Process first frame
    height, width = frame_dimensions
    
    # Clear CUDA cache before processing
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Process depth for first frame if it's a keyframe
    if is_keyframe:
        depth = model.infer_image(first_frame, input_size=input_size)
    else:
        depth = None  # Will be interpolated later
    
    # Process ball detection for first frame
    if first_frame_idx in ball_results_by_frame_id:
        if depth is not None:
            all_depths[first_frame_idx] = depth
        
        ball_results = ball_results_by_frame_id[first_frame_idx]
        
        for ball_result in ball_results:
            ball_x = int(ball_result.x_frame)
            ball_y = int(ball_result.y_frame)
            timestamp = ball_result.timestamp
            
            # Get normalized coordinates for the resized depth map
            norm_x = min(int(ball_x * width / orig_width), width - 1)
            norm_y = min(int(ball_y * height / orig_height), height - 1)
            
            # Get depth at ball position if available - using circular mean
            ball_z = None
            if depth is not None:
                # Instead of single pixel depth[norm_y, norm_x], use circular mean
                ball_z = get_circular_mean_depth(depth, norm_x, norm_y, radius=ball_depth_radius)
            
            # Create 3D ball info
            ball_info = BallInfo3D(ball_x, ball_y, ball_z, timestamp, interpolated=(not is_keyframe))
            ball_info_3d_by_frame[first_frame_idx] = ball_info
            previous_ball_info = ball_info
    
    # Update progress bar
    progress_bar.update(1)
    
    # Process the rest of the frames
    for frame, frame_id, _, _, is_keyframe in frame_generator:
        # Store frame data
        all_frame_data[frame_id] = {'is_keyframe': is_keyframe}
        
        # Process depth for current frame if it's a keyframe
        if is_keyframe:
            depth = model.infer_image(frame, input_size=input_size)
        else:
            depth = None  # Will be interpolated later
        
        # Process ball detection for current frame if available
        if frame_id in ball_results_by_frame_id:
            if depth is not None:
                all_depths[frame_id] = depth
            
            ball_results = ball_results_by_frame_id[frame_id]
            
            for ball_result in ball_results:
                ball_x = int(ball_result.x_frame)
                ball_y = int(ball_result.y_frame)
                timestamp = ball_result.timestamp
                
                # Get normalized coordinates for the resized depth map
                norm_x = min(int(ball_x * width / orig_width), width - 1)
                norm_y = min(int(ball_y * height / orig_height), height - 1)
                
                # Get depth at ball position if available - using circular mean
                ball_z = None
                if depth is not None:
                    # Instead of single pixel depth[norm_y, norm_x], use circular mean
                    ball_z = get_circular_mean_depth(depth, norm_x, norm_y, radius=ball_depth_radius)
                
                # Create 3D ball info
                ball_info = BallInfo3D(ball_x, ball_y, ball_z, timestamp, interpolated=(not is_keyframe))
                ball_info_3d_by_frame[frame_id] = ball_info
                
                # Only calculate speed if we have previous ball data with actual Z values
                if previous_ball_info is not None and previous_ball_info.z is not None and ball_z is not None:
                    ball_info.calculate_speed(previous_ball_info)
                
                previous_ball_info = ball_info
        
        # Update progress bar
        progress_bar.update(1)
        
        # Free memory every 20 frames
        if frame_id % 20 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
    
    # Close progress bar
    progress_bar.close()
    
    # Get all unique frame IDs in ascending order
    all_frame_ids = sorted(all_frame_data.keys())
    
    # Interpolate Z values for frames where we skipped depth inference
    print("Interpolating depth values for skipped frames...")
    ball_info_3d_by_frame = interpolate_z_values(ball_info_3d_by_frame, all_frame_ids)
    
    # Calculate speeds for all interpolated frames
    print("Calculating speeds for interpolated frames...")
    frame_ids = sorted(ball_info_3d_by_frame.keys())
    prev_ball_info = None
    
    for frame_id in frame_ids:
        ball_info = ball_info_3d_by_frame[frame_id]
        if prev_ball_info is not None and ball_info.z is not None and prev_ball_info.z is not None:
            ball_info.calculate_speed(prev_ball_info)
        prev_ball_info = ball_info
    
    print(f"Processed depths for {len(all_depths)} keyframes with ball detections")
    print(f"Extracted and interpolated 3D ball information for {len(ball_info_3d_by_frame)} frames")
    
    # Detect ball hits
    print("Detecting ball hits...")
    hit_frames = detect_hits(ball_info_3d_by_frame, fps=video_fps)
    print(f"Detected {len(hit_frames)} hits at frames: {hit_frames}")
    
    return {
        'all_depths': all_depths,
        'ball_info_3d_by_frame': ball_info_3d_by_frame,
        'hit_frames': hit_frames,
        'frame_dimensions': frame_dimensions,
        'video_fps': video_fps,
        'total_frames': total_frames_count,
        'all_frame_data': all_frame_data
    }

# Function to create the visualization video
def create_visualization_video(video_path, inference_results, ball_results_by_frame_id, orig_dimensions, params, output_path):
    print("\n" + "="*50)
    print("PHASE 2: VIDEO VISUALIZATION")
    print("="*50)
    
    max_len = params.get('max_len', -1)
    target_fps = params.get('target_fps', -1)
    max_res = params.get('max_res', 960)
    
    # Extract results from inference
    all_depths = inference_results['all_depths']
    ball_info_3d_by_frame = inference_results['ball_info_3d_by_frame']
    hit_frames = inference_results['hit_frames']
    height, width = inference_results['frame_dimensions']
    video_fps = inference_results['video_fps']
    total_frames = inference_results['total_frames']
    all_frame_data = inference_results['all_frame_data']
    
    orig_width, orig_height = orig_dimensions
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
    video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    if not video_writer.isOpened():
        raise Exception(f"Failed to open VideoWriter for {output_path}")
    
    print(f"Creating visualization video with dimensions {width}x{height} @ {video_fps}fps")
    
    # Reset ball history for visualization
    ball_history = []
    max_history = 30  # Maximum number of past positions to display
    
    # Create a cache for interpolated depth maps
    depth_map_cache = {}
    
    # Second pass: Create visualization with hit detection
    frame_generator = read_video_frames_generator(video_path, max_len, target_fps, max_res, 0)  # No skipping for visualization
    
    # Initialize progress bar for visualization
    progress_bar = tqdm(total=total_frames, desc="Video visualization")
    
    # Process all frames for visualization
    for frame, frame_id, _, _, _ in frame_generator:
        # Convert the frame to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get depth map - either from all_depths or interpolate
        if frame_id in all_depths:
            # Use the pre-computed depth map
            depth_map = all_depths[frame_id]
            depth_map_cache[frame_id] = depth_map
        else:
            # Use interpolated depth map if we have ball info with interpolated Z
            if frame_id in ball_info_3d_by_frame and ball_info_3d_by_frame[frame_id].interpolated:
                # Check if we've already calculated this depth map
                if frame_id in depth_map_cache:
                    depth_map = depth_map_cache[frame_id]
                else:
                    # Find the nearest keyframes with depth maps
                    all_depth_frames = sorted(all_depths.keys())
                    if len(all_depth_frames) >= 2:
                        # Find the nearest depth frames before and after
                        prev_frame = max([f for f in all_depth_frames if f < frame_id], default=None)
                        next_frame = min([f for f in all_depth_frames if f > frame_id], default=None)
                        
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
                        elif prev_frame is not None:
                            depth_map = all_depths[prev_frame]
                            depth_map_cache[frame_id] = depth_map
                        elif next_frame is not None:
                            depth_map = all_depths[next_frame]
                            depth_map_cache[frame_id] = depth_map
                        else:
                            depth_map = None
                    else:
                        depth_map = None
            else:
                depth_map = None
        
        # Apply color to depth map for visualization if available
        if depth_map is not None:
            # Normalize and colorize depth map
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # Resize depth map to match frame size if needed
            if depth_colored.shape[:2] != (frame_bgr.shape[0], frame_bgr.shape[1]):
                depth_colored = cv2.resize(depth_colored, (frame_bgr.shape[1], frame_bgr.shape[0]))
            
            # Blend depth map with original frame
            blended = cv2.addWeighted(frame_bgr, 0.7, depth_colored, 0.3, 0)
        else:
            blended = frame_bgr
        
        # Draw ball trajectory history
        for prev_x, prev_y in ball_history:
            cv2.circle(blended, (int(prev_x * width / orig_width), 
                               int(prev_y * height / orig_height)), 
                     radius=2, color=(0, 255, 255), thickness=-1)
        
        # Draw current ball position
        if frame_id in ball_results_by_frame_id:
            ball_results = ball_results_by_frame_id[frame_id]
            for ball_result in ball_results:
                # Scale ball coordinates to match the frame size
                ball_x = int(ball_result.x_frame * width / orig_width)
                ball_y = int(ball_result.y_frame * height / orig_height)
                
                # Add to history
                ball_history.append((ball_result.x_frame, ball_result.y_frame))
                if len(ball_history) > max_history:
                    ball_history.pop(0)  # Remove oldest position
                
                # Draw current ball
                cv2.circle(blended, (ball_x, ball_y), radius=10, color=(0, 255, 255), thickness=2)
                
                # If this is a hit frame, highlight it
                if frame_id in hit_frames:
                    # Draw a larger circle for hits with different color
                    cv2.circle(blended, (ball_x, ball_y), radius=20, color=(0, 0, 255), thickness=3)
                    
                    # Add "HIT" text
                    cv2.putText(blended, "HIT", (ball_x+25, ball_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # If the ball has depth info, show it
                if frame_id in ball_info_3d_by_frame:
                    ball_info = ball_info_3d_by_frame[frame_id]
                    
                    # Mark interpolated depth values differently
                    depth_text = f"Z: {ball_info.z:.2f}" 
                    if ball_info.interpolated:
                        depth_text += " (interp)"
                        text_color = (100, 255, 100)  # Light green for interpolated
                    else:
                        text_color = (255, 255, 255)  # White for direct measurement
                    
                    cv2.putText(blended, depth_text, (ball_x+15, ball_y+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    if ball_info.speed_3d is not None:
                        speed_text = f"Speed: {ball_info.speed_3d:.2f}"
                        cv2.putText(blended, speed_text, (ball_x+15, ball_y+45), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Write the frame to video
        video_writer.write(blended)
        
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

def load_model(model_type, device):
    """Load the Depth-Anything-V2 model"""
    # Add the Depth-Anything-V2 repository to Python path
    repo_path = os.path.join(os.getcwd(), "Depth-Anything-V2")
    sys.path.append(repo_path)
    
    # Import the DepthAnythingV2 model
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        print("Successfully imported Depth-Anything-V2 modules")
    except Exception as e:
        print(f"Error importing Depth-Anything-V2 modules: {e}")
        sys.exit(1)
        
    # Model configurations from Depth-Anything-V2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if model_type not in model_configs:
        print(f"Unknown model type: {model_type}. Using 'vitl' instead.")
        model_type = 'vits'
    
    print(f"Using model: {model_type}")
    
    # Check for the checkpoint
    checkpoint_path = f'./checkpoints/depth_anything_v2_{model_type}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        print(f"Please download the model from: https://huggingface.co/depth-anything/Depth-Anything-V2-{model_type.replace('vit', '').upper()}/resolve/main/depth_anything_v2_{model_type}.pth?download=true")
        sys.exit(1)
    else:
        print(f"Found checkpoint at {checkpoint_path}")
    
    try:
        # Initialize the depth model
        print("Initializing model...")
        model = DepthAnythingV2(**model_configs[model_type])
        print("Loading model weights...")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        model = model.to(device).eval()
        print("Model successfully loaded")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_ball_data(ball_json, camera_id):
    """Load ball detection data from JSON file"""
    # Load ball detection results
    try:
        print(f"Loading ball detections from {ball_json}...")
        with open(ball_json) as f:
            json_data = json.load(f)
    
        ball_results_by_frame_id = defaultdict(list)
        for dets_per_frame in json_data:
            for cur_det_per_frame in dets_per_frame:
                if cur_det_per_frame and cur_det_per_frame['camera_id'] == camera_id and cur_det_per_frame['detection_class'] == 'm_ball':
                    obj = DetectionResult(**cur_det_per_frame)
                    frame_id = obj.frame_id
                    ball_results_by_frame_id[frame_id].append(obj)
        
        print(f"Loaded ball detections for {len(ball_results_by_frame_id)} frames")
        return ball_results_by_frame_id
    except Exception as e:
        print(f"Error loading ball detections: {e}")
        sys.exit(1)

def main():
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
    parser.add_argument('--max_res', type=int, default=960,
                        help='Maximum resolution for processing')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input size for depth model')
    parser.add_argument('--skip_frames', type=int, default=11,
                        help='Number of frames to skip (0 = no skipping)')
    parser.add_argument('--max_len', type=int, default=-1,
                        help='Maximum number of frames to process (-1 = all)')
    parser.add_argument('--target_fps', type=float, default=-1,
                        help='Target FPS for output video (-1 = original)')
    parser.add_argument('--no_interpolation', action='store_true',
                        help='Disable depth interpolation')
    parser.add_argument('--ball_depth_radius', type=int, default=10,
                        help='Radius (in pixels) for circular area to calculate mean ball depth')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video file not found at {args.video_path}")
        sys.exit(1)
    if not os.path.exists(args.ball_json):
        print(f"ERROR: Ball JSON file not found at {args.ball_json}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device - use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model
    model = load_model(args.model_type, device)
    
    # Load ball data
    ball_results_by_frame_id = load_ball_data(args.ball_json, args.camera_id)
    
    # Read original video dimensions
    cap = cv2.VideoCapture(args.video_path)
    orig_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Original video dimensions: {orig_frame_width}x{orig_frame_height}, FPS: {orig_fps}, Total frames: {orig_total_frames}")
    
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
    
    try:
        # PHASE 1: Depth inference and hit detection with frame skipping
        inference_results = run_depth_inference(
            args.video_path, 
            ball_results_by_frame_id, 
            (orig_frame_width, orig_frame_height),
            model,
            processing_params
        )
        
        # PHASE 2: Video visualization
        output_path = create_visualization_video(
            args.video_path,
            inference_results,
            ball_results_by_frame_id,
            (orig_frame_width, orig_frame_height),
            processing_params,
            output_video_avi
        )
        
        # Save the 3D ball data to a file
        print("\n" + "="*50)
        print("PHASE 3: SAVING BALL DATA")
        print("="*50)
        
        print("Saving 3D ball data...")
        ball_data = []
        for frame_id, ball_info in inference_results['ball_info_3d_by_frame'].items():
            is_hit = frame_id in inference_results['hit_frames']
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
    
        print(f"3D ball data with hit detection saved to {output_json_path}")
        
        # Final summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Processed {inference_results['total_frames']} frames")
        print(f"Detected {len(inference_results['hit_frames'])} ball hits")
        print(f"Output files:")
        print(f"- AVI video: {output_video_avi}")
        print(f"- Ball data: {output_json_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
