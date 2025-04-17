#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
import requests
import json
import logging
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_with_minimal_processing(image_path, conf_threshold=0.01):
    """
    Test the API with minimal processing and filtering
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        # Check if server is running
        try:
            ping_response = requests.get("http://localhost:8000/api/v1/ping", timeout=3)
            if ping_response.status_code != 200:
                logger.error(f"Server not responding properly: {ping_response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API server not running: {str(e)}")
            logger.info("Please start the server with 'uvicorn app.main:app --reload'")
            return None

        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Test /detect endpoint with minimal processing
        logger.info(f"Testing /detect endpoint with minimal processing")

        files = {'car_image': (os.path.basename(image_path), image_data, 'image/jpeg')}
        data = {
            'return_type': 'json',
            'conf_threshold': conf_threshold,  # Use very low threshold
            'skip_enhancement': True  # Skip image enhancement
        }

        start_time = time.time()
        response = requests.post(
            'http://localhost:8000/api/v1/detect',
            files=files,
            data=data
        )
        api_time = time.time() - start_time

        logger.info(f"API response time: {api_time:.2f} seconds")

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None

        result = response.json()

        # Save original detection results
        detection_results = {
            "image_path": image_path,
            "api_response_time": api_time,
            "confidence_threshold": conf_threshold,
            "detection_found": result.get("detection_found", False),
            "confidence": result.get("confidence"),
            "bounding_box": result.get("bounding_box")
        }

        # Get the annotated image
        if result.get("detection_found", False):
            # Test again with image return type
            files = {'car_image': (os.path.basename(image_path), image_data, 'image/jpeg')}
            data = {
                'return_type': 'image',
                'conf_threshold': conf_threshold,
                'skip_enhancement': True
            }

            img_response = requests.post(
                'http://localhost:8000/api/v1/detect',
                files=files,
                data=data
            )

            if img_response.status_code == 200:
                # Save the annotated image
                output_filename = f"minimal_proc_{os.path.basename(image_path)}"
                with open(output_filename, 'wb') as f:
                    f.write(img_response.content)
                logger.info(f"Saved annotated image to {output_filename}")
                detection_results["annotated_image"] = output_filename

        return detection_results

    except Exception as e:
        logger.error(f"Error in API test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_multiple_images(image_paths=None, conf_threshold=0.01):
    """
    Test multiple images with minimal processing
    """
    if not image_paths:
        # Use default images from the Dataset folder
        dataset_dir = Path("Dataset")
        image_paths = [str(p) for p in list(dataset_dir.glob("*.jpeg"))[:10]]

    if not image_paths:
        logger.error("No images to process")
        return

    results = []
    success_count = 0

    for path in image_paths:
        logger.info(f"=== Testing image: {path} ===")
        result = test_with_minimal_processing(path, conf_threshold)

        if result:
            results.append(result)
            if result.get("detection_found", False):
                success_count += 1
                logger.info(f"Detection found with confidence: {result['confidence']:.4f}")
            else:
                logger.info("No detection found")

    # Print summary
    total = len(image_paths)
    success_rate = success_count / total * 100 if total > 0 else 0
    logger.info(f"=== Summary ===")
    logger.info(f"Tested {total} images")
    logger.info(f"Successful detections: {success_count}/{total} ({success_rate:.1f}%)")

    # Save results to JSON file
    with open("minimal_processing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to minimal_processing_results.json")

    return results

def main():
    """
    Main entry point
    """
    conf_threshold = 0.01  # Very low confidence threshold

    # Check if image paths are provided
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Use default paths
        image_paths = None

    # Run tests
    test_multiple_images(image_paths, conf_threshold)

if __name__ == "__main__":
    main()