#!/usr/bin/env python
"""
License Plate Detection API - Runner Script

This script starts the FastAPI server with the correct Python import paths.
Instead of running from the app directory (which causes import issues),
run this script from the project root:

    python run.py

or:

    python -m uvicorn app.main:app --reload --port 8000

"""
import os
import sys
import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model_file():
    """Check if model file exists and report status"""
    model_path = Path("app/models/best.pt")
    if not model_path.exists():
        print("\n⚠️ WARNING: Model file not found at", model_path)
        print("The API will run but detection will fail until you add a model file.")
        print("You need to add a YOLOv8 model trained on license plates to this location.")
        return False

    # Check model file size to verify it's likely a valid model
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"Model file found: {model_path} (Size: {model_size_mb:.2f} MB)")

    if model_size_mb < 1.0:
        logger.warning(f"Model file seems very small ({model_size_mb:.2f} MB). This may not be a valid model.")

    return True

def create_required_directories():
    """Create all required directories for the API to function properly"""
    # Models directory
    os.makedirs("app/models", exist_ok=True)

    # Output directory (as defined in settings.OUTPUT_DIR)
    os.makedirs("app/static/output", exist_ok=True)

    # Media directories (as defined in settings.MEDIA_ROOT)
    os.makedirs("app/static/media", exist_ok=True)
    os.makedirs("app/static/media/processed", exist_ok=True)

    logger.info("Created all required directories")

def main():
    """Main entry point for the API server"""
    # Check if model file exists
    check_model_file()

    # Create required directories
    create_required_directories()

    # Start the FastAPI server with the correct import path
    logger.info("Starting License Plate Detection API server...")

    # Set port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))

    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()