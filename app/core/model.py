from functools import lru_cache
import torch
from ultralytics import YOLO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_model(model_path: str) -> YOLO:
    """
    Load and cache the YOLO model.

    Args:
        model_path: Path to the model file

    Returns:
        YOLO model instance

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(str(model_path), task='detect')

        # Disable training-related features
        model.trainer = None
        model.ckpt = None

        # Warm up the model
        logger.info("Warming up model...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            model(dummy_input)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def get_model(model_path: str = "app/models/best.pt") -> YOLO:
    """
    Get the cached YOLO model instance.

    Args:
        model_path: Path to the model file

    Returns:
        YOLO model instance

    Raises:
        RuntimeError: If model file doesn't exist or loading fails
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")

    return load_model(str(model_path))

# Initialize model at startup
model = get_model()