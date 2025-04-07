import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence

import cv2
import numpy as np
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_FILENAME = "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx"
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_KEYPOINT_RADIUS = 5
PADDING_VALUE = 114  # Padding value for the preprocessed image

# Colors for visualization
WRIST_COLOR = (0, 0, 255)  # Red (in BGR)
KEYPOINT_COLOR = (0, 255, 0)  # Green (in BGR)

# COCO keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Indices of wrist keypoints
WRIST_INDICES = [9, 10]  # left_wrist, right_wrist

def check_model_exists(model_path: Optional[str] = None) -> Optional[str]:
    """
    Check if the RTMO model exists at the specified path or in default locations.
    
    Args:
        model_path: Path to the ONNX model file, or None to search in default locations
        
    Returns:
        Path to the model file if found, None otherwise
    """
    # If model_path is provided, check if it exists
    if model_path is not None:
        if os.path.exists(model_path):
            logger.info(f"Found model at specified path: {model_path}")
            return model_path
        else:
            logger.warning(f"Model not found at specified path: {model_path}")
    
    # Try default location in current directory
    default_path = os.path.join("checkpoints", DEFAULT_MODEL_FILENAME)
    if os.path.exists(default_path):
        logger.info(f"Found model at default path: {default_path}")
        return default_path
        
    # Check if model exists in common locations
    common_locations = [
        os.path.join(os.getcwd(), default_path),
        os.path.join(os.path.expanduser('~/.cache/rtmlib'), 'hub/checkpoints', DEFAULT_MODEL_FILENAME),
    ]
    
    for loc in common_locations:
        if os.path.exists(loc):
            logger.info(f"Found model at: {loc}")
            return loc
    
    # Model not found in any location        
    logger.error(f"Model not found in any standard location: {DEFAULT_MODEL_FILENAME}")
    return None

class RTMO:
    """
    RTMO model for human pose estimation.
    
    This class provides functionalities for loading and running the RTMO model
    for pose estimation, with specific focus on detecting wrists for tennis
    hit analysis.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        device: str = 'cuda'
    ):
        """
        Initialize the RTMO model.
        
        Args:
            model_path: Path to the ONNX model file, or None to search in default locations
            input_size: Input size for the model as (width, height)
            device: Device to run inference on ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If the model file cannot be found
            RuntimeError: If the model cannot be loaded
        """
        # Check if model exists or find it
        found_model_path = check_model_exists(model_path)
        
        if found_model_path is None:
            msg = f"RTMO model file not found. Please ensure the model file '{DEFAULT_MODEL_FILENAME}' exists."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        # Use CUDA if available and requested, otherwise fallback to CPU
        available_providers = ort.get_available_providers()
        provider = ('CUDAExecutionProvider' 
                   if device == 'cuda' and 'CUDAExecutionProvider' in available_providers 
                   else 'CPUExecutionProvider')
        
        try:
            self.session = ort.InferenceSession(
                path_or_bytes=found_model_path,
                providers=[provider]
            )
            
            self.input_size = input_size
            logger.info(f'Loaded RTMO model from {found_model_path} using {provider}')
        except Exception as e:
            msg = f"Failed to load RTMO model: {e}"
            logger.error(msg)
            raise RuntimeError(msg)
        
        # Define COCO keypoint names and wrist indices
        self.keypoint_names = KEYPOINT_NAMES
        self.wrist_indices = WRIST_INDICES

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on an image.
        
        Args:
            img: Input image in BGR format (HWC)
            
        Returns:
            Tuple containing:
                - keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
                - scores: Array of shape (num_persons, 17) with keypoint confidence scores
        """
        img, ratio = self.preprocess(img)
        outputs = self.inference(img)
        keypoints, scores = self.postprocess(outputs, ratio)
        return keypoints, scores

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for inference.
        
        Args:
            img: Input image in BGR format (HWC)
            
        Returns:
            Tuple containing:
                - padded_img: Preprocessed image ready for model input
                - ratio: Scaling ratio applied during preprocessing
        """
        # Create a padded image of the target size
        padded_img = np.ones(
            (self.input_size[0], self.input_size[1], 3),
            dtype=np.uint8
        ) * PADDING_VALUE

        # Calculate the scaling ratio
        ratio = min(
            self.input_size[0] / img.shape[0],
            self.input_size[1] / img.shape[1]
        )
        
        # Resize the image while maintaining aspect ratio
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Place the resized image in the padded image
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio

    def inference(self, img: np.ndarray) -> List:
        """
        Run model inference.
        
        Args:
            img: Preprocessed image
            
        Returns:
            List of model output tensors
        """
        # Convert image to model input format
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32)
        input_data = img[None, :, :, :]  # Add batch dimension

        # Run inference
        inputs = {self.session.get_inputs()[0].name: input_data}
        outputs = self.session.run(None, inputs)

        return outputs

    def postprocess(self, 
                   outputs: List, 
                   ratio: float = 1.0, 
                   confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process model outputs.
        
        Args:
            outputs: Model outputs from inference
            ratio: Scaling ratio from preprocessing
            confidence_threshold: Minimum confidence score for detection
            
        Returns:
            Tuple containing:
                - keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
                - scores: Array of shape (num_persons, 17) with keypoint confidence scores
        """
        det_outputs, pose_outputs = outputs
        
        # Apply confidence threshold to filter detections
        final_scores = det_outputs[0, :, 4]
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
            logger.debug("No persons detected in the image")
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

        logger.debug(f"Detected {len(keypoints)} persons")
        return keypoints, scores
        
    def get_wrist_positions(self, 
                           keypoints: np.ndarray, 
                           scores: np.ndarray, 
                           min_score: float = DEFAULT_CONFIDENCE_THRESHOLD
                           ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract wrist positions from keypoints.
        
        Args:
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
            min_score: Minimum confidence score to consider a wrist keypoint valid
            
        Returns:
            Tuple containing:
                - wrist_positions: List of valid wrist positions (x, y) for all persons
                - wrist_scores: List of confidence scores for the valid wrist positions
        """
        wrist_positions = []
        wrist_scores = []
        
        for person_idx in range(keypoints.shape[0]):
            for wrist_idx in self.wrist_indices:
                if scores[person_idx, wrist_idx] >= min_score:
                    wrist_positions.append(keypoints[person_idx, wrist_idx])
                    wrist_scores.append(scores[person_idx, wrist_idx])
        
        logger.debug(f"Found {len(wrist_positions)} valid wrists")
        return wrist_positions, wrist_scores

    def visualize_poses(self, 
                     img: np.ndarray, 
                     keypoints: np.ndarray, 
                     scores: np.ndarray,
                     min_score: float = DEFAULT_CONFIDENCE_THRESHOLD,
                     keypoint_radius: int = DEFAULT_KEYPOINT_RADIUS,
                     draw_connections: bool = True
                     ) -> np.ndarray:
        """
        Draw detected poses on the image.
        
        Args:
            img: Input image in BGR format
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
            min_score: Minimum confidence score to visualize a keypoint
            keypoint_radius: Radius of the circles drawn for keypoints
            draw_connections: Whether to draw connections between keypoints
            
        Returns:
            Image with visualized poses
        """
        # Make a copy of the image to avoid modifying the original
        vis_img = img.copy()
        
        # Define connections between keypoints for visualization
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw each person's pose
        for person_idx in range(keypoints.shape[0]):
            # Draw connections first so keypoints are on top
            if draw_connections:
                for conn in connections:
                    pt1 = keypoints[person_idx, conn[0]]
                    pt2 = keypoints[person_idx, conn[1]]
                    
                    # Only draw if both points are visible
                    if (scores[person_idx, conn[0]] > min_score and 
                        scores[person_idx, conn[1]] > min_score):
                        
                        pt1 = (int(pt1[0]), int(pt1[1]))
                        pt2 = (int(pt2[0]), int(pt2[1]))
                        cv2.line(vis_img, pt1, pt2, (0, 255, 255), 2)
            
            # Draw keypoints
            for kp_idx in range(keypoints.shape[1]):
                x, y = keypoints[person_idx, kp_idx]
                conf = scores[person_idx, kp_idx]
                
                if conf > min_score:
                    # Different color for wrists vs other keypoints
                    color = WRIST_COLOR if kp_idx in self.wrist_indices else KEYPOINT_COLOR
                    cv2.circle(vis_img, (int(x), int(y)), keypoint_radius, color, -1)
        
        return vis_img
    
    def get_formatted_results(self, 
                            keypoints: np.ndarray, 
                            scores: np.ndarray,
                            min_score: float = DEFAULT_CONFIDENCE_THRESHOLD
                            ) -> List[Dict[str, Any]]:
        """
        Format detection results into a more convenient structure.
        
        Args:
            keypoints: Array of shape (num_persons, 17, 2) with keypoint coordinates
            scores: Array of shape (num_persons, 17) with keypoint confidence scores
            min_score: Minimum confidence score to include a keypoint
            
        Returns:
            List of dictionaries, one per person, containing keypoint information
        """
        results = []
        
        for person_idx in range(keypoints.shape[0]):
            person_keypoints = {}
            valid_keypoints = 0
            
            for kp_idx, kp_name in enumerate(self.keypoint_names):
                x, y = keypoints[person_idx, kp_idx]
                conf = float(scores[person_idx, kp_idx])
                
                if conf > min_score:
                    person_keypoints[kp_name] = {
                        'x': float(x),
                        'y': float(y),
                        'confidence': conf,
                        'is_wrist': kp_idx in self.wrist_indices
                    }
                    valid_keypoints += 1
            
            # Only include persons with at least one valid keypoint
            if valid_keypoints > 0:
                results.append({
                    'keypoints': person_keypoints,
                    'valid_keypoint_count': valid_keypoints,
                    'wrists': [person_keypoints[kp_name] for kp_name in ['left_wrist', 'right_wrist'] 
                               if kp_name in person_keypoints]
                })
        
        return results

# Test the model if running this script directly
if __name__ == "__main__":
    import argparse
    
    def main() -> None:
        """
        Main function to test the RTMO model on an image.
        """
        # Configure argument parser
        parser = argparse.ArgumentParser(description='Test RTMO model on an image')
        parser.add_argument('--image', type=str, required=True, help='Path to test image')
        parser.add_argument('--model', type=str, default=None, help='Path to RTMO model')
        parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run inference on')
        parser.add_argument('--output', type=str, default="rtmo_output.jpg", help='Output image path')
        parser.add_argument('--json-output', type=str, default=None, help='Path to save JSON results (optional)')
        parser.add_argument('--draw-connections', action='store_true', help='Draw connections between keypoints')
        parser.add_argument('--min-score', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, 
                            help=f'Minimum confidence score (default: {DEFAULT_CONFIDENCE_THRESHOLD})')
        parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                            help='Logging level')
        
        args = parser.parse_args()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        try:
            # Validate input file
            if not os.path.exists(args.image):
                raise FileNotFoundError(f"Image not found at {args.image}")
            
            logger.info(f"Testing RTMO model on image: {args.image}")
            
            # Initialize model
            rtmo = RTMO(model_path=args.model, device=args.device)
            
            # Load and process image
            img = cv2.imread(args.image)
            if img is None:
                raise RuntimeError(f"Failed to load image: {args.image}")
                
            logger.info("Running inference...")
            keypoints, scores = rtmo(img)
            
            # Get wrist positions
            wrist_positions, wrist_scores = rtmo.get_wrist_positions(keypoints, scores, min_score=args.min_score)
            
            # Log results
            logger.info(f"Detected {len(keypoints)} persons")
            logger.info(f"Found {len(wrist_positions)} valid wrists")
            
            # Create visualization
            logger.info("Creating visualization...")
            vis_img = rtmo.visualize_poses(
                img, keypoints, scores, 
                min_score=args.min_score, 
                draw_connections=args.draw_connections
            )
            
            # Save visualization
            cv2.imwrite(args.output, vis_img)
            logger.info(f"Visualization saved to {args.output}")
            
            # Get and save formatted results
            if args.json_output:
                import json
                
                # Get formatted results
                formatted_results = rtmo.get_formatted_results(keypoints, scores, min_score=args.min_score)
                
                # Add metadata
                output_data = {
                    'metadata': {
                        'image_path': os.path.abspath(args.image),
                        'image_dimensions': img.shape[:2],
                        'model': 'RTMO',
                        'persons_detected': len(keypoints),
                        'wrists_detected': len(wrist_positions)
                    },
                    'persons': formatted_results
                }
                
                # Save to JSON
                with open(args.json_output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                    
                logger.info(f"Results saved to {args.json_output}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    main()
