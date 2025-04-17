import requests
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import sys
import glob
import time
import base64
from PIL import Image

def check_model_file():
    """Check if model file exists and report status"""
    model_path = Path("app/models/best.pt")
    if not model_path.exists():
        print("\n⚠️ WARNING: Model file not found at", model_path)
        print("The API will run but detection will fail until you add a model file.")
        print("You need to add a YOLOv8 model trained on license plates to this location.")

        # Check for model files in other locations
        model_files = []
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".pt"):
                    model_files.append(os.path.join(root, file))

        if model_files:
            print("\nFound model files in other locations:")
            for model_file in model_files:
                print(f"  - {model_file}")

            # Ask if user wants to move one of these files
            print("\nDo you want to copy one of these model files to app/models/best.pt? (y/n)")
            choice = input().lower()
            if choice.startswith('y'):
                print("Enter the number of the model to use:")
                for i, model_file in enumerate(model_files):
                    print(f"  {i+1}. {model_file}")

                try:
                    model_index = int(input()) - 1
                    if 0 <= model_index < len(model_files):
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        shutil.copy(model_files[model_index], model_path)
                        print(f"✅ Copied {model_files[model_index]} to {model_path}")
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")

        return False

    # Check file size to ensure it's a valid model
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n✅ Model file found at {model_path} (Size: {file_size_mb:.2f} MB)")

    # Check model file date
    import datetime
    mod_time = os.path.getmtime(model_path)
    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Last modified: {mod_time_str}")

    return True

def test_ping():
    """Test the ping endpoint."""
    url = "http://localhost:8000/api/v1/ping"
    print(f"\n🔍 Testing ping endpoint: {url}")

    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        end_time = time.time()

        print(f"✅ Status Code: {response.status_code}")
        print(f"⏱️ Response Time: {(end_time - start_time):.2f} seconds")

        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"📋 Response: {json.dumps(response.json(), indent=2)}")
        else:
            print("📋 Response: [Non-JSON content]")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Failed to connect to server. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_detect():
    """Test the license plate detection endpoint."""
    url = "http://localhost:8000/api/v1/detect"
    print(f"\n🔍 Testing license plate detection endpoint: {url}")

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

    # Display available images
    print("\nAvailable images:")
    for i, img_path in enumerate(image_files):
        img_size = os.path.getsize(img_path) / 1024  # Size in KB
        print(f"{i+1}. {os.path.basename(img_path)} ({img_size:.1f} KB)")

    # Ask user to select an image
    while True:
        selection = input("\nSelect an image number (or 'q' to quit): ")
        if selection.lower() == 'q':
            return

        try:
            idx = int(selection) - 1
            if 0 <= idx < len(image_files):
                selected_image = image_files[idx]
                break
            else:
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(image_files)}.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # Get image dimensions
    try:
        with Image.open(selected_image) as img:
            width, height = img.size
            print(f"\n📷 Selected image: {os.path.basename(selected_image)}")
            print(f"📐 Dimensions: {width}x{height} pixels")
            print(f"📊 Size: {os.path.getsize(selected_image)/1024:.1f} KB")
    except Exception as e:
        print(f"⚠️ Warning: Could not get image dimensions: {e}")

    # Set up form data
    try:
        # Allow user to set confidence threshold
        conf_input = input("\nEnter confidence threshold (0.0-1.0) or press Enter for default: ")
        conf_threshold = 0.4  # Default value
        if conf_input:
            try:
                conf_threshold = float(conf_input)
                if conf_threshold < 0 or conf_threshold > 1:
                    print("⚠️ Invalid confidence value. Using default 0.4")
                    conf_threshold = 0.4
            except ValueError:
                print("⚠️ Invalid confidence value. Using default 0.4")

        # Choose return type
        return_type_input = input("Return as [j]son or [i]mage? (default: json): ")
        return_type = "json"
        if return_type_input.lower() in ['i', 'image']:
            return_type = "image"

        # Set up the form data for the request
        files = {
            'car_image': (os.path.basename(selected_image), open(selected_image, 'rb'))
        }
        data = {
            'conf_threshold': str(conf_threshold),
            'return_type': return_type
        }

        # Get user input for custom text (optional)
        custom_text = input("Enter custom text (optional): ")
        if custom_text:
            data['custom_text'] = custom_text

        print(f"\n🚀 Sending request with: confidence={conf_threshold}, return_type={return_type}")
        start_time = time.time()
        response = requests.post(url, files=files, data=data, timeout=30)
        end_time = time.time()

        print(f"✅ Status Code: {response.status_code}")
        print(f"⏱️ Response Time: {(end_time - start_time):.2f} seconds")

        response_data = response.json()

        # Process the response
        if return_type == "json":
            print(f"📋 Response: {json.dumps(response_data, indent=2)}")
        else:
            # Handle image response
            detection_data = response_data.get('detection', {})
            print(f"📋 Detection Data: {json.dumps(detection_data, indent=2)}")

            # Save the image if it exists
            if 'image' in response_data:
                try:
                    img_data = base64.b64decode(response_data['image'])
                    output_path = f"output_{int(time.time())}.jpg"
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    print(f"🖼️ Annotated image saved to: {output_path}")
                except Exception as e:
                    print(f"❌ Error saving image: {str(e)}")

    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out after 30 seconds")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Failed to connect to server. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def check_model_files():
    """Check if model files exist and report their sizes."""
    print("\n🔍 Checking model files...")

    model_paths = [
        "models/best.pt",
        "models/yolov8n.pt"
    ]

    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {path}: Found ({size_mb:.1f} MB)")
        else:
            print(f"❌ {path}: Not found")

def main():
    """Run the API tests."""
    print("\n" + "=" * 60)
    print("🚗 License Plate Detection API Tester")
    print("=" * 60)

    # Check if model files exist
    check_model_files()

    # Test the ping endpoint
    test_ping()

    # Test the detect endpoint
    test_detect()

    print("\n" + "=" * 60)
    print("✅ Testing completed")
    print("=" * 60)

if __name__ == "__main__":
    main()