from functools import lru_cache
import torch
from ultralytics import YOLO
import logging
from pathlib import Path
import numpy as np
import cv2
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model
global_model = None

def enhance_image(image):
    """
    Apply image enhancement techniques to improve license plate visibility

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

        # Apply enhancements conditionally based on image metrics
        # If image is too dark, apply brightness enhancement
        if brightness < 80:
            logger.info("Applying brightness enhancement")
            # Use CLAHE on the value channel of HSV
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # If image has low contrast, enhance it
        if contrast < 50:
            logger.info("Applying contrast enhancement")
            # Use CLAHE on the value channel of HSV
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # If image is noisy, apply noise reduction
        if noise_level > 500:
            logger.info("Applying noise reduction")
            # Use bilateral filter for noise reduction while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply sharpening to all images to enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced
    except Exception as e:
        logger.error(f"Error in image enhancement: {str(e)}")
        return image  # Return original image if enhancement fails

def preprocess_image(image):
    """
    Preprocess image for model inference

    Args:
        image: Image as numpy array (BGR format)

    Returns:
        Preprocessed image ready for model inference
    """
    try:
        if image is None:
            logger.error("Received None image in preprocess_image")
            return None

        # Log original image shape and type
        logger.info(f"Original image shape: {image.shape}, dtype: {image.dtype}")

        # Check image validity
        if image.size == 0 or len(image.shape) < 2:
            logger.error(f"Invalid image shape: {image.shape}")
            return None

        # Store original dimensions for reference
        original_height, original_width = image.shape[:2]

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            logger.info("Converting grayscale image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Convert BGR to RGB (YOLOv8 expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply image enhancement techniques
        enhanced_image = enhance_image(image)
        if enhanced_image is not None:
            # Use enhanced image if successful
            image = enhanced_image
            logger.info("Successfully applied image enhancement")

        # Resize to YOLOv8 preferred input size (640x640)
        # while preserving aspect ratio
        target_size = 640

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

        return square_img

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

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
    Get singleton model instance

    Returns:
        YOLO model instance
    """
    global global_model

    if global_model is None:
        logger.info("Loading model (first-time initialization)")
        global_model = load_model()

    return global_model

# Initialize model at startup
try:
    model = get_model()
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize model at startup: {str(e)}")
    raise