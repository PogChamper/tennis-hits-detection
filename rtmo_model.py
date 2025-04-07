import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import os
import sys

def check_model_exists(model_path=None):
    """
    Check if the model exists at the specified path or in the default location.
    
    Args:
        model_path: Path to the ONNX model file, or None to use default
        
    Returns:
        Path to the model file if found, None otherwise
    """
    # If model_path is provided, check if it exists
    if model_path is not None and os.path.exists(model_path):
        return model_path
    
    # Try default location in current directory
    default_path = "checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx"
    if os.path.exists(default_path):
        return default_path
        
    # Check if model exists in common locations
    common_locations = [
        os.path.join(os.getcwd(), default_path),
        os.path.join(os.path.expanduser('~/.cache/rtmlib'), 'hub/checkpoints', default_path),
    ]
    
    for loc in common_locations:
        if os.path.exists(loc):
            return loc
            
    return None

class RTMO:
    """RTMO model for human pose estimation"""
    
    def __init__(
        self,
        model_path: str = None,
        input_size: tuple = (640, 640),
        device: str = 'cuda'
    ):
        """
        Initialize the RTMO model.
        
        Args:
            model_path: Path to the ONNX model file
            input_size: Input size for the model (width, height)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        # Check if model exists or find it
        found_model_path = check_model_exists(model_path)
        
        if found_model_path is None:
            print("ERROR: RTMO model file not found.")
            print(f"Please ensure the model file 'rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx' exists in the current directory")
            print(f"Current directory: {os.getcwd()}")
            sys.exit(1)
        
        # Use CUDA if available and requested, otherwise fallback to CPU
        provider = 'CUDAExecutionProvider' if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers() else 'CPUExecutionProvider'
        
        try:
            self.session = ort.InferenceSession(
                path_or_bytes=found_model_path,
                providers=[provider]
            )
            
            self.input_size = input_size
            print(f'Loaded RTMO model from {found_model_path} using {provider}')
        except Exception as e:
            print(f"ERROR: Failed to load RTMO model: {e}")
            sys.exit(1)
        
        # Define COCO keypoint names for reference
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Indices of keypoints that are most relevant for tennis hit detection
        self.wrist_indices = [9, 10]  # left_wrist, right_wrist

    def __call__(self, img: np.ndarray):
        """
        Run inference on an image.
        
        Args:
            img: Input image in BGR format (HWC)
            
        Returns:
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
        """
        img, ratio = self.preprocess(img)
        outputs = self.inference(img)
        keypoints, scores = self.postprocess(outputs, ratio)
        return keypoints, scores

    def preprocess(self, img: np.ndarray):
        """
        Preprocess image for inference.
        
        Args:
            img: Input image in BGR format (HWC)
            
        Returns:
            padded_img: Preprocessed image
            ratio: Scaling ratio
        """
        padded_img = np.ones(
            (self.input_size[0], self.input_size[1], 3),
            dtype=np.uint8
        ) * 114

        ratio = min(
            self.input_size[0] / img.shape[0],
            self.input_size[1] / img.shape[1]
        )
        
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR
        )
        
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio

    def inference(self, img: np.ndarray):
        """
        Run model inference.
        
        Args:
            img: Preprocessed image
            
        Returns:
            outputs: Model outputs
        """
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32)
        input_data = img[None, :, :, :]  # Add batch dimension

        inputs = {self.session.get_inputs()[0].name: input_data}
        outputs = self.session.run(None, inputs)

        return outputs

    def postprocess(self, outputs: list, ratio: float = 1.0):
        """
        Process model outputs.
        
        Args:
            outputs: Model outputs
            ratio: Scaling ratio from preprocessing
            
        Returns:
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
        """
        det_outputs, pose_outputs = outputs
        
        final_scores = det_outputs[0, :, 4]
        confidence_threshold = 0.3
        isscore = final_scores > confidence_threshold
        isbbox = [i for i in isscore]

        # Get keypoints and scores for detected persons
        keypoints = pose_outputs[0, :, :, :2] / ratio
        scores = pose_outputs[0, :, :, 2]

        # Filter by detection score
        keypoints = keypoints[isbbox]
        scores = scores[isbbox]

        # If no detections, return empty arrays
        if len(keypoints) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

        return keypoints, scores
        
    def get_wrist_positions(self, keypoints, scores, min_score=0.3):
        """
        Extract wrist positions from keypoints.
        
        Args:
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
            min_score: Minimum confidence score to consider a wrist keypoint valid
            
        Returns:
            wrist_positions: List of valid wrist positions (x, y) for all persons
            wrist_scores: List of confidence scores for the valid wrist positions
        """
        wrist_positions = []
        wrist_scores = []
        
        for person_idx in range(keypoints.shape[0]):
            for wrist_idx in self.wrist_indices:
                if scores[person_idx, wrist_idx] >= min_score:
                    wrist_positions.append(keypoints[person_idx, wrist_idx])
                    wrist_scores.append(scores[person_idx, wrist_idx])
        
        return wrist_positions, wrist_scores

# Test the model if running this script directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RTMO model on an image')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--model', type=str, default=None, help='Path to RTMO model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cpu or cuda)')
    
    args = parser.parse_args()
    
    if args.image is None:
        print("Please provide an image path with --image")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Image not found at {args.image}")
        sys.exit(1)
    
    # Initialize model
    rtmo = RTMO(model_path=args.model, device=args.device)
    
    # Load and process image
    img = cv2.imread(args.image)
    keypoints, scores = rtmo(img)
    
    # Get wrist positions
    wrist_positions, wrist_scores = rtmo.get_wrist_positions(keypoints, scores)
    
    # Visualize results
    print(f"Detected {len(keypoints)} persons")
    print(f"Found {len(wrist_positions)} valid wrists")
    
    # Draw keypoints on image
    for person_idx in range(keypoints.shape[0]):
        for kp_idx in range(keypoints.shape[1]):
            x, y = keypoints[person_idx, kp_idx]
            conf = scores[person_idx, kp_idx]
            
            if conf > 0.3:
                # Different color for wrists (red) vs other keypoints (green)
                color = (0, 0, 255) if kp_idx in rtmo.wrist_indices else (0, 255, 0)
                cv2.circle(img, (int(x), int(y)), 5, color, -1)
    
    cv2.imwrite("rtmo_output.jpg", img)
    print("Visualization saved to rtmo_output.jpg") 