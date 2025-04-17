import os
import requests
import time
import json
import random
import cv2
import numpy as np
from pathlib import Path
import sys

# API endpoints
API_URL = "http://localhost:8000/api/v1"
DETECT_ENDPOINT = f"{API_URL}/detect"

def test_detection_parameters(image_path, conf_threshold, iou_threshold, skip_enhancement):
    """Test license plate detection with specific parameters"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return None

        # Prepare form data
        with open(image_path, "rb") as f:
            files = {"car_image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "conf_threshold": str(conf_threshold),
                "return_type": "json",
                "skip_enhancement": str(skip_enhancement),
                "iou_threshold": str(iou_threshold)
            }

            print(f"Testing with conf={conf_threshold}, iou={iou_threshold}, skip_enhancement={skip_enhancement}")

            # Make API call
            start_time = time.time()
            response = requests.post(DETECT_ENDPOINT, files=files, data=data)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                detection_found = result.get('detection_found', False)
                confidence = result.get('confidence', 0)

                print(f"  - Detection: {detection_found}, Confidence: {confidence:.4f if confidence else 0}, Time: {elapsed:.2f}s")
                return result
            else:
                print(f"  - Error: {response.status_code}, {response.text}")
                return None
    except Exception as e:
        print(f"  - Exception: {str(e)}")
        return None

def start_server():
    """Start the API server if not already running"""
    try:
        # Check if server is running
        response = requests.get(f"{API_URL}/ping", timeout=2)
        if response.status_code == 200:
            print("Server is already running")
            return True
    except:
        print("Starting server...")
        # You could add code here to start the server programmatically
        print("Please start the server manually and try again")
        return False

    return True

def get_sample_images(count=3):
    """Get sample images from Dataset folder"""
    dataset_path = Path("Dataset")
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"Error: Dataset folder not found at {dataset_path}")
        return []

    image_files = list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.jpg"))
    if not image_files:
        print("No image files found in Dataset folder")
        return []

    # Limit to requested count
    if len(image_files) > count:
        return random.sample(image_files, count)
    return image_files

def test_single_image(image_path):
    """Test a single image with various parameters"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return

    print(f"Testing single image: {image_path}")

    # Define parameter combinations to test
    configurations = [
        # Default current settings
        {"conf": 0.01, "iou": 0.1, "skip_enhancement": False},

        # Test with enhancement disabled
        {"conf": 0.01, "iou": 0.1, "skip_enhancement": True},

        # Higher confidence
        {"conf": 0.05, "iou": 0.1, "skip_enhancement": False},
        {"conf": 0.15, "iou": 0.1, "skip_enhancement": False},

        # Higher IOU
        {"conf": 0.01, "iou": 0.2, "skip_enhancement": False},
        {"conf": 0.01, "iou": 0.3, "skip_enhancement": False},
    ]

    best_config = None
    best_confidence = 0

    for config in configurations:
        print(f"\nTesting configuration: {config}")
        result = test_detection_parameters(
            image_path,
            config["conf"],
            config["iou"],
            config["skip_enhancement"]
        )

        if result and result.get("detection_found", False):
            confidence = result.get("confidence", 0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_config = config

    print("\n=== SUMMARY FOR SINGLE IMAGE ===")
    if best_config:
        print(f"Best configuration: {best_config}")
        print(f"Best confidence: {best_confidence:.4f}")
    else:
        print("No successful detections")

def main():
    """Main test function"""
    print("=== LICENSE PLATE DETECTION PARAMETER TEST ===")

    # Check if server is running
    if not start_server():
        return

    # Check if a specific image is provided
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
        return

    # Get sample images
    test_images = get_sample_images(count=5)
    if not test_images:
        return

    print(f"Testing with {len(test_images)} images from Dataset folder")

    # Define parameter combinations to test
    configurations = [
        # Default current settings
        {"conf": 0.01, "iou": 0.1, "skip_enhancement": False},

        # Test with enhancement disabled
        {"conf": 0.01, "iou": 0.1, "skip_enhancement": True},

        # Higher confidence thresholds
        {"conf": 0.05, "iou": 0.1, "skip_enhancement": False},
        {"conf": 0.15, "iou": 0.1, "skip_enhancement": False},

        # Higher IOU thresholds
        {"conf": 0.01, "iou": 0.2, "skip_enhancement": False},
        {"conf": 0.01, "iou": 0.3, "skip_enhancement": False},

        # Combined variations
        {"conf": 0.05, "iou": 0.2, "skip_enhancement": False},
        {"conf": 0.05, "iou": 0.2, "skip_enhancement": True},
    ]

    # Track best results
    best_config = None
    best_detection_count = 0
    best_avg_confidence = 0

    results = []

    # Test each configuration on all images
    for config in configurations:
        print(f"\nTesting configuration: {config}")
        config_results = {
            "config": config,
            "detections": 0,
            "total_confidence": 0,
            "image_results": []
        }

        for img_path in test_images:
            print(f"Testing image: {img_path}")
            result = test_detection_parameters(
                str(img_path),
                config["conf"],
                config["iou"],
                config["skip_enhancement"]
            )

            # Record results
            if result:
                detection_found = result.get("detection_found", False)
                confidence = result.get("confidence", 0) if detection_found else 0

                img_result = {
                    "image": str(img_path),
                    "detection_found": detection_found,
                    "confidence": confidence
                }
                config_results["image_results"].append(img_result)

                if detection_found:
                    config_results["detections"] += 1
                    config_results["total_confidence"] += confidence
            else:
                # Record failed test
                img_result = {
                    "image": str(img_path),
                    "detection_found": False,
                    "confidence": 0,
                    "error": True
                }
                config_results["image_results"].append(img_result)

        # Calculate average confidence
        avg_confidence = 0
        if config_results["detections"] > 0:
            avg_confidence = config_results["total_confidence"] / config_results["detections"]

        print(f"Configuration results: {config_results['detections']}/{len(test_images)} detections, avg confidence: {avg_confidence:.4f}")

        config_results["avg_confidence"] = avg_confidence
        results.append(config_results)

        # Check if this is the best configuration
        if config_results["detections"] > best_detection_count or (
            config_results["detections"] == best_detection_count and
            avg_confidence > best_avg_confidence
        ):
            best_config = config
            best_detection_count = config_results["detections"]
            best_avg_confidence = avg_confidence

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Best configuration: {best_config}")
    print(f"Best detection count: {best_detection_count}/{len(test_images)}")
    print(f"Best average confidence: {best_avg_confidence:.4f}")

    # Save results
    with open("parameter_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to parameter_test_results.json")

if __name__ == "__main__":
    main()