# Tennis Analysis

Tennis analysis task using depth estimation and player pose detection for detection ball hits.

## Project Structure
```tennis-analysis/
├── data/
│ ├── videos/ # Input tennis videos
│ └── jsons/ball/ # Ball detection JSON files
├── checkpoints/ # Model weights
│ ├── depth_anything_v2_vitl.pth # Depth estimation model
│ └── rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx # RTMO pose model
├── outputs/ # Output files
├── Dockerfile # Docker configuration
├── infer_depth.py # Depth inference script
├── detect_hits_from_poses.py # Pose-based hit detection script
└── rtmo_model.py # RTMO model wrapper
```

## Getting Started

### 1. Download Required Models

#### Depth-Anything-V2 Models
Download the depth estimation models from the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) repository.


```bash
mkdir -p checkpoints
```

### For vitl model (recommended)
```bash
wget -O checkpoints/depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```
### For vitb model (smaller but faster)
```bash
wget -O checkpoints/depth_anything_v2_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
```

#### RTMO Model
Download the RTMO model from the [rtmlib](https://github.com/Tau-J/rtmlib) repository:

```bash
wget -O checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx https://huggingface.co/pesi/rtmo/resolve/main/rtmo-l.onnx
```

### 2. Build Docker Image

```bash
docker build -t tennis-analysis .
```

### 3. Prepare Your Data

Place your tennis videos in the `data/videos/` directory and ball detection JSON files in the `data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/` directory.

### 4. Run Analysis

#### On Windows

**3D Depth Analysis:**
```cmd
docker run --gpus all -v %CD%/data:/app/data -v %CD%/outputs:/app/outputs -v %CD%/checkpoints:/app/checkpoints --entrypoint python3 tennis-analysis infer_depth.py --video_path data/videos/025958_6m47s_7m13s.mp4 --ball_json data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json --output_dir outputs --model_type vitl
```

**Pose-Based Hit Detection:**
```cmd
docker run --gpus all -v %CD%/data:/app/data -v %CD%/outputs:/app/outputs -v %CD%/checkpoints:/app/checkpoints --entrypoint python3 tennis-analysis detect_hits_from_poses.py --video_path data/videos/025958_6m47s_7m13s.mp4 --ball_json data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json --output_dir outputs --rtmo_model checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx
```

#### On Linux

**3D Depth Analysis:**
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python3 tennis-analysis infer_depth.py --video_path data/videos/025958_6m47s_7m13s.mp4 --ball_json data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json --output_dir outputs --model_type vitl
```

**Pose-Based Hit Detection:**
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python3 tennis-analysis detect_hits_from_poses.py --video_path data/videos/025958_6m47s_7m13s.mp4 --ball_json data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json --output_dir outputs --rtmo_model checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx
```

## Command-Line Arguments

### infer_depth.py
```
--video_path Path to the tennis video file
--ball_json Path to the ball detection JSON file
--camera_id Camera ID in the ball detection data (default: camera1)
--output_dir Directory to save outputs (default: outputs)
--model_type Model type to use (choices: vits, vitb, vitl, vitg; default: vitl)
--max_res Maximum resolution for processing (default: 960)
--input_size Input size for depth model (default: 518)
--skip_frames Number of frames to skip (0 = no skipping; default: 11)
--max_len Maximum number of frames to process (-1 = all; default: -1)
--target_fps Target FPS for output video (-1 = original; default: -1)
--no_interpolation Disable depth interpolation
--ball_depth_radius Radius for circular area to calculate mean ball depth (default: 10)
```

### detect_hits_from_poses.py
```
--video_path Path to the tennis video file
--ball_json Path to the ball detection JSON file
--camera_id Camera ID in the ball detection data (default: camera1)
--output_dir Directory to save outputs (default: outputs)
--rtmo_model Path to RTMO model file
--device Device to run inference on (cpu or cuda; default: cuda)
--hit_distance Maximum distance between wrist and ball to consider a hit (default: 120)
--min_frames_between_hits Minimum number of frames between consecutive hits (default: 15)
--max_frames Maximum number of frames to process (-1 = all; default: -1)
```

## Outputs

The analysis generates the following outputs in the specified output directory:

- **AVI Video**: Visualization of ball tracking with depth information or pose detection
- **JSON File**: Data containing ball trajectory, depth information, and detected hits

## Acknowledgments

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for depth estimation models
- [rtmlib](https://github.com/Tau-J/rtmlib) for RTMO pose estimation models
