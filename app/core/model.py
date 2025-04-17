from functools import lru_cache
import torch
from ultralytics import YOLO
import logging
from pathlib import Path
import numpy as np
import cv2
import os
import sys
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model and lock for thread safety
global_model = None
model_lock = threading.Lock()

def enhance_image(image):
    """
    Apply image enhancement techniques with less aggressive processing
    to better preserve license plate features.

    Args:
        image: Image to enhance (RGB format)

    Returns:
        Enhanced image
    """
    try:
        # Make a copy of the input image
        enhanced = image.copy()

        # Calculate image quality metrics
        # Convert to grayscale for analysis only
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate contrast as standard deviation of grayscale pixels
        contrast = np.std(gray)

        # Calculate brightness as mean of grayscale pixels
        brightness = np.mean(gray)

        # Calculate noise level (using Laplacian)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

        logger.info(f"Image metrics - contrast: {contrast:.2f}, brightness: {brightness:.2f}, noise: {noise_level:.2f}")

        # Apply enhancements more selectively and with milder settings
        # Only apply brightness enhancement if image is very dark
        if brightness < 50:  # More strict threshold (was 80)
            logger.info("Applying mild brightness enhancement")
            # Use CLAHE on the value channel of HSV with milder settings
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # Reduced clip limit from 3.0 to 2.0
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Only apply contrast enhancement if contrast is very low
        if contrast < 40:  # More strict threshold (was 60)
            logger.info("Applying mild contrast enhancement")
            # Use CLAHE on the value channel of HSV with milder settings
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # Reduced clip limit from 3.0 to 2.0
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Only apply noise reduction if image is very noisy
        if noise_level > 1000:  # More strict threshold (was 500)
            logger.info("Applying mild noise reduction")
            # Use bilateral filter with smaller kernel for less aggressive smoothing
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)  # Was (9, 75, 75)

        # Apply milder sharpening to all images to enhance edges
        # Changed from [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]] to:
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Removed the adaptive thresholding and weighted combination steps
        # that were potentially destroying license plate features

        return enhanced
    except Exception as e:
        logger.error(f"Error in image enhancement: {str(e)}")
        return image  # Return original image if enhancement fails

def preprocess_image(image, skip_enhancement=False):
    """
    Preprocess image for model inference

    Args:
        image: Image as numpy array (BGR format)
        skip_enhancement: If True, skip the enhance_image step to avoid double enhancement

    Returns:
        Tuple of (preprocessed_image, transform_params) where transform_params is a dict with:
        - scale: scaling factor applied
        - x_offset: x offset in the square canvas
        - y_offset: y offset in the square canvas
        - original_size: original image dimensions (height, width)
    """
    try:
        if image is None:
            logger.error("Received None image in preprocess_image")
            return None, None

        # Log original image shape and type
        logger.info(f"Original image shape: {image.shape}, dtype: {image.dtype}")

        # Check image validity
        if image.size == 0 or len(image.shape) < 2:
            logger.error(f"Invalid image shape: {image.shape}")
            return None, None

        # Store original dimensions for reference
        original_height, original_width = image.shape[:2]

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            logger.info("Converting grayscale image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Convert BGR to RGB (YOLOv8 expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply image enhancement techniques only if not skipped
        if not skip_enhancement:
            enhanced_image = enhance_image(image)
            if enhanced_image is not None:
                # Use enhanced image if successful
                image = enhanced_image
                logger.info("Successfully applied image enhancement")
        else:
            logger.info("Skipping image enhancement as requested")

        # Resize to YOLOv8 input size (1024x1024) for better resolution
        # while preserving aspect ratio
        target_size = 1024

        # Calculate scale to maintain aspect ratio
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a square canvas with black padding
        square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Center the resized image on the square canvas
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        # Log preprocessing results
        logger.info(f"Preprocessed image shape: {square_img.shape}")

        # Create transform parameters dictionary for coordinate mapping
        transform_params = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "original_size": (original_height, original_width)
        }

        return square_img, transform_params

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def load_model(model_path=None):
    """
    Load YOLOv8 model for license plate detection

    Args:
        model_path: Path to the YOLOv8 model file (default: None)

    Returns:
        YOLO model instance or None if loading fails
    """
    if model_path is None:
        # Default model path
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")

    model_path = os.path.abspath(model_path)
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None

    try:
        # Log model file size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {model_size_mb:.2f} MB")

        # Load model with low confidence threshold to catch all potential detections
        # We'll filter by confidence in the API
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_model():
    """
    Get singleton model instance with thread safety.
    Uses a lock to prevent race conditions when multiple threads try to load the model.

    Returns:
        YOLO model instance or None if loading fails
    """
    global global_model

    # Fast path - model already loaded
    if global_model is not None:
        return global_model

    # Lock for thread safety when loading the model
    with model_lock:
        # Double-check inside the lock to avoid race conditions
        if global_model is None:
            logger.info("Loading model (first-time initialization)")
            global_model = load_model()

            if global_model is None:
                logger.error("Failed to load model - will retry on next request")
                return None

            logger.info("Model loaded successfully and cached for future use")

        return global_model

def map_coordinates_to_original(box, transform_params):
    """
    Map coordinates from processed image space (1024x1024) back to original image space

    Args:
        box: List or array [x1, y1, x2, y2] in processed image space
        transform_params: Dictionary with transformation parameters:
            - scale: scaling factor applied during preprocessing
            - x_offset: x offset in the square canvas
            - y_offset: y offset in the square canvas
            - original_size: original image dimensions (height, width)

    Returns:
        List [x1, y1, x2, y2] in original image space
    """
    try:
        # Unpack parameters
        scale = transform_params["scale"]
        x_offset = transform_params["x_offset"]
        y_offset = transform_params["y_offset"]

        # Unpack box coordinates
        x1, y1, x2, y2 = box

        # Adjust for padding offsets
        x1 = x1 - x_offset
        y1 = y1 - y_offset
        x2 = x2 - x_offset
        y2 = y2 - y_offset

        # Convert back to original scale
        x1 = max(0, int(x1 / scale))
        y1 = max(0, int(y1 / scale))
        x2 = max(0, int(x2 / scale))
        y2 = max(0, int(y2 / scale))

        return [x1, y1, x2, y2]
    except Exception as e:
        logger.error(f"Error mapping coordinates: {str(e)}")
        logger.error(f"Transform params: {transform_params}")
        logger.error(f"Box: {box}")
        import traceback
        logger.error(traceback.format_exc())
        # Return input box as fallback
        return box