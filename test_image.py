import os
import requests
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def test_image(image_path):
    """Test the updated license plate detection API with a single image"""
    print(f"=== TESTING IMAGE: {image_path} ===")

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return

    # Test /detect endpoint
    print("\nTesting with /detect endpoint:")
    try:
        with open(image_path, 'rb') as f:
            files = {'car_image': f}
            data = {'return_type': 'json', 'conf_threshold': 0.2}
            response = requests.post('http://localhost:8000/api/v1/detect', files=files, data=data)

            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Detection found: {result['detection_found']}")
                if result['detection_found']:
                    print(f"Confidence: {result['confidence']:.4f}")
                    box = result['bounding_box']
                    print(f"Bounding box: {box}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error with /detect endpoint: {str(e)}")

    # Test /detect-and-process endpoint
    print("\nTesting with /detect-and-process endpoint:")
    try:
        with open(image_path, 'rb') as f:
            files = {'car_image': f}
            data = {
                'return_type': 'json',
                'conf_threshold': 0.2,
                'custom_text': 'TEST'
            }
            response = requests.post('http://localhost:8000/api/v1/detect-and-process', files=files, data=data)

            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Detection found: {result['detection']['detection_found']}")
                if result['detection']['detection_found']:
                    print(f"Confidence: {result['detection']['confidence']:.4f}")
                    print(f"Total detections: {len(result['detection']['detections'])}")

                    # Save the annotated image if available
                    if 'image' in result:
                        img_data = base64.b64decode(result['image'])
                        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_annotated.jpg"
                        with open(output_filename, 'wb') as img_file:
                            img_file.write(img_data)
                        print(f"Saved annotated image to {output_filename}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error with /detect-and-process endpoint: {str(e)}")

def main():
    # Get image path from command line argument or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset/20.jpeg"
    test_image(image_path)

if __name__ == "__main__":
    main()