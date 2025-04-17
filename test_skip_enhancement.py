import os
import requests
import time
import sys
import json
import cv2
import numpy as np

def create_test_image():
    """Create a synthetic test image with a rectangular area simulating a license plate"""
    if not os.path.exists("test_car.jpg"):
        print("Creating test image...")
        # Create a simple image with a rectangle simulating a license plate
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Fill with gray (simulating a car)
        img[:, :] = (120, 120, 120)

        # Add a white rectangle (simulating a license plate)
        plate_x1, plate_y1 = 250, 200
        plate_x2, plate_y2 = 400, 250
        img[plate_y1:plate_y2, plate_x1:plate_x2] = (255, 255, 255)

        # Add some text to the plate
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "ABC123", (plate_x1 + 10, plate_y1 + 35), font, 1, (0, 0, 0), 2)

        # Add some noise to make it more realistic
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        # Save the image
        cv2.imwrite("test_car.jpg", img)
        print("Test image created.")
    else:
        print("Using existing test_car.jpg")

    return "test_car.jpg"

def test_detection_endpoint(skip_enhancement):
    """Test the /detect endpoint with skip_enhancement parameter"""
    url = "http://localhost:8000/api/v1/detect"

    image_path = create_test_image()

    with open(image_path, "rb") as f:
        files = {"car_image": ("test_car.jpg", f, "image/jpeg")}
        data = {
            "conf_threshold": "0.15",
            "return_type": "json",
            "skip_enhancement": str(skip_enhancement)
        }

        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time

        print(f"Detection with skip_enhancement={skip_enhancement} took {elapsed:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print(f"Detection found: {result.get('detection_found', False)}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Processing time: {result.get('processing_time')}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

def test_detect_and_process_endpoint(skip_enhancement):
    """Test the /detect-and-process endpoint with skip_enhancement parameter"""
    url = "http://localhost:8000/api/v1/detect-and-process"

    image_path = create_test_image()

    with open(image_path, "rb") as f:
        files = {"car_image": ("test_car.jpg", f, "image/jpeg")}
        data = {
            "conf_threshold": "0.15",
            "return_type": "json",
            "custom_text": "HIDDEN",
            "skip_enhancement": str(skip_enhancement)
        }

        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time

        print(f"Detect & Process with skip_enhancement={skip_enhancement} took {elapsed:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print(f"Detection found: {result.get('detection', {}).get('detection_found', False)}")
            print(f"Confidence: {result.get('detection', {}).get('confidence')}")
            print(f"Detection time: {result.get('detection', {}).get('detection_time')}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

def main():
    print("TESTING /DETECT ENDPOINT")
    print("-----------------------")

    # Test with enhancement
    print("\nTesting with enhancement (skip_enhancement=False):")
    with_enhancement = test_detection_endpoint(False)

    # Test without enhancement
    print("\nTesting without enhancement (skip_enhancement=True):")
    without_enhancement = test_detection_endpoint(True)

    print("\n\nTESTING /DETECT-AND-PROCESS ENDPOINT")
    print("------------------------------------")

    # Test with enhancement
    print("\nTesting with enhancement (skip_enhancement=False):")
    process_with_enhancement = test_detect_and_process_endpoint(False)

    # Test without enhancement
    print("\nTesting without enhancement (skip_enhancement=True):")
    process_without_enhancement = test_detect_and_process_endpoint(True)

    # Compare results
    print("\n\nRESULT COMPARISON")
    print("----------------")

    if with_enhancement and without_enhancement:
        print(f"/detect - With enhancement confidence: {with_enhancement.get('confidence')}")
        print(f"/detect - Without enhancement confidence: {without_enhancement.get('confidence')}")

        confidence_diff = (
            without_enhancement.get('confidence', 0) -
            with_enhancement.get('confidence', 0)
        )

        if confidence_diff > 0:
            print(f"Skipping enhancement improved confidence by {confidence_diff:.4f}")
        else:
            print(f"Skipping enhancement decreased confidence by {abs(confidence_diff):.4f}")

    if process_with_enhancement and process_without_enhancement:
        with_conf = process_with_enhancement.get('detection', {}).get('confidence', 0)
        without_conf = process_without_enhancement.get('detection', {}).get('confidence', 0)

        print(f"/detect-and-process - With enhancement confidence: {with_conf}")
        print(f"/detect-and-process - Without enhancement confidence: {without_conf}")

        confidence_diff = without_conf - with_conf

        if confidence_diff > 0:
            print(f"Skipping enhancement improved confidence by {confidence_diff:.4f}")
        else:
            print(f"Skipping enhancement decreased confidence by {abs(confidence_diff):.4f}")

if __name__ == "__main__":
    main()