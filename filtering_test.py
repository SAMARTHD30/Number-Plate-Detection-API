import requests
import json
import os
from pathlib import Path
import cv2
import numpy as np
import base64
from PIL import Image
import time

def test_detection_with_new_filters():
    """Test the license plate detection with new filtering parameters"""
    url = "http://localhost:8000/api/v1/detect"
    print(f"\n🔍 Testing license plate detection with new filters: {url}")

    # Get list of all image files in the Dataset directory
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext}")))

    if not image_files:
        print(f"❌ Error: No image files found in '{dataset_dir}'.")
        return

    # Test each image
    for idx, img_path in enumerate(image_files[:5]):  # Test first 5 images
        print(f"\n[{idx+1}/{len(image_files[:5])}] Testing image: {os.path.basename(img_path)}")

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
            print(f"📐 Dimensions: {width}x{height} pixels")
            print(f"📊 Size: {os.path.getsize(img_path)/1024:.1f} KB")

        # Set up form data with our new confidence threshold of 0.2
        files = {
            'car_image': (os.path.basename(img_path), open(img_path, 'rb'))
        }
        data = {
            'conf_threshold': '0.2',  # Our new threshold
            'return_type': 'json'
        }

        try:
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=30)
            end_time = time.time()

            print(f"✅ Status Code: {response.status_code}")
            print(f"⏱️ Response Time: {(end_time - start_time):.2f} seconds")

            if response.status_code == 200:
                response_data = response.json()

                # Check if license plate was detected
                if response_data.get('detection_found', False):
                    confidence = response_data.get('confidence', 0)
                    box = response_data.get('bounding_box', {})
                    print(f"✅ License plate detected with confidence: {confidence:.4f}")
                    print(f"📍 Bounding box: {box}")
                else:
                    print("❌ No license plate detected")
            else:
                print(f"❌ Error: {response.text}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import glob  # Import here for proper scope
    test_detection_with_new_filters()