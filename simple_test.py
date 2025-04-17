import requests
import os
import json
import time
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000/api/v1/detect"

def test_image(image_path):
    """Test the license plate detection with an image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return

    print(f"Testing image: {image_path}")

    with open(image_path, "rb") as f:
        files = {"car_image": (os.path.basename(image_path), f, "image/jpeg")}
        data = {
            "conf_threshold": "0.01",
            "iou_threshold": "0.1",
            "skip_enhancement": "False",
            "return_type": "json"
        }

        try:
            start_time = time.time()
            response = requests.post(API_URL, files=files, data=data)
            elapsed = time.time() - start_time

            print(f"Request time: {elapsed:.2f} seconds")
            print(f"Status code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {str(e)}")

def main():
    print("Simple API Test")

    # Test ping endpoint first
    try:
        ping_response = requests.get("http://localhost:8000/api/v1/ping")
        if ping_response.status_code == 200:
            print("API is running")
        else:
            print(f"API ping failed: {ping_response.status_code}")
            return
    except Exception as e:
        print(f"API connection error: {str(e)}")
        return

    # Get all image files in the Dataset folder
    dataset_path = Path("Dataset")
    image_files = list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.jpg"))

    if not image_files:
        print("No image files found in Dataset folder")
        return

    # Test the first few images
    for i, image_path in enumerate(image_files[:3]):
        print(f"\n--- Test {i+1} ---")
        test_image(str(image_path))
        time.sleep(1)  # Add a small delay between requests

if __name__ == "__main__":
    main()