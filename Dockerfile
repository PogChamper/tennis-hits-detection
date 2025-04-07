# Use NVIDIA CUDA base image - using a verified tag
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /app

# Copy the scripts
COPY infer_depth.py detect_hits_from_poses.py rtmo_model.py ./

# Create directories for data
RUN mkdir -p data/videos data/jsons/ball/unet_mbnv2_norelu6_576x1024_11ch_our_data_blob_det checkpoints outputs

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    tqdm \
    scipy \
    pydantic \
    onnxruntime-gpu

# Clone Depth-Anything-V2 repository
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2 && \
    cd Depth-Anything-V2 && \
    pip3 install -r requirements.txt

# Create a directory for the RTMO model
RUN mkdir -p /app/models

# Set up volume mount points for data
VOLUME ["/app/data", "/app/checkpoints", "/app/outputs", "/app/models"]

# Set the entrypoint script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]