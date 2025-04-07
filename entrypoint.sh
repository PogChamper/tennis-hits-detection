#!/bin/bash
set -e

if [ "$1" = "depth" ]; then
    # Run infer_depth.py
    python3 infer_depth.py \
        --video_path "${2:-data/videos/025958_6m47s_7m13s.mp4}" \
        --ball_json "${3:-data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json}" \
        --camera_id "${4:-camera1}" \
        --output_dir "${5:-outputs}" \
        --model_type "${6:-vitb}" \
        --skip_frames "${7:-11}" \
        --ball_depth_radius "${8:-10}"

elif [ "$1" = "pose" ]; then
    # Run detect_hits_from_poses.py
    python3 detect_hits_from_poses.py \
        --video_path "${2:-data/videos/025958_6m47s_7m13s.mp4}" \
        --ball_json "${3:-data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det/025958_6m47s_7m13s.json}" \
        --camera_id "${4:-camera1}" \
        --output_dir "${5:-outputs}" \
        --rtmo_model "${6:-/app/models/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx}" \
        --device "${7:-cuda}" \
        --hit_distance "${8:-120}"

else
    echo "Usage: docker run [options] <image> [depth|pose] [args...]"
    echo ""
    echo "For depth inference:"
    echo "  docker run [options] <image> depth [video_path] [ball_json] [camera_id] [output_dir] [model_type] [skip_frames] [ball_depth_radius]"
    echo ""
    echo "For pose-based hit detection:"
    echo "  docker run [options] <image> pose [video_path] [ball_json] [camera_id] [output_dir] [rtmo_model] [device] [hit_distance]"
    exit 1
fi