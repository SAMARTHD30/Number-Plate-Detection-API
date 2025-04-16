# License Plate Detection API Testing Guide

This document provides instructions for testing all endpoints of the License Plate Detection API.

## Prerequisites
- The API server is running (`python -m uvicorn app.main:app --reload`)
- You have images containing license plates for testing
- You have a tool for making HTTP requests (Swagger UI, curl, or Postman)

## Base URL
```
http://localhost:8000
```

## API Endpoints

### 1. Root Endpoint
Check if the API is running.

```bash
GET /
```

Expected Response:
```json
{
    "message": "License Plate Detection API is running"
}
```

### 2. Health Check
Check API status.

```bash
GET /api/v1/ping
```

Expected Response:
```json
{
    "status": "ok",
    "message": "API is running",
    "version": "1.0.0"
}
```

### 3. License Plate Detection
Detect license plates in an image.

```bash
POST /api/v1/detect
```

Parameters:
- `file`: Image file (required)
- `return_type`: "json" or "image" (default: "json")
- `custom_text`: Optional text to add to the image

#### Test with curl:
```bash
# JSON response
curl -X POST -F "file=@path/to/image.jpg" -F "return_type=json" http://localhost:8000/api/v1/detect

# Image response
curl -X POST -F "file=@path/to/image.jpg" -F "return_type=image" http://localhost:8000/api/v1/detect --output result.jpg
```

Expected JSON Response:
```json
{
    "detection_found": true,
    "plate_text": "ABC123",
    "confidence": 0.95,
    "bounding_box": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
    }
}
```

### 4. Process and Return Image
Process an image and return the result.

```bash
POST /api/v1/process-image
```

Parameters:
- `car_image`: Image file (required)
- `custom_text`: Optional text to add to the image
- `return_type`: "json" or "image" (default: "image")

#### Test with curl:
```bash
# JSON response
curl -X POST -F "car_image=@path/to/image.jpg" -F "return_type=json" -F "custom_text=Test Text" http://localhost:8000/api/v1/process-image

# Image response
curl -X POST -F "car_image=@path/to/image.jpg" -F "return_type=image" http://localhost:8000/api/v1/process-image --output processed.jpg
```

Expected JSON Response:
```json
{
    "image_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "success",
    "custom_text": "Test Text",
    "message": "Image processed successfully",
    "image_url": "/api/v1/image/550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-03-14T12:00:00.000Z",
    "image_data": {
        "content_type": "image/jpeg",
        "base64_data": "data:image/jpeg;base64,..."
    }
}
```

### 5. Get Processed Image
Retrieve a processed image by ID.

```bash
GET /api/v1/image/{image_id}
```

Parameters:
- `image_id`: UUID of the processed image

#### Test with curl:
```bash
curl http://localhost:8000/api/v1/image/550e8400-e29b-41d4-a716-446655440000 --output image.jpg
```

## Testing with Python

Here's a Python script to test the API:

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("Root Response:", response.json())

def test_ping():
    response = requests.get(f"{BASE_URL}/api/v1/ping")
    print("Ping Response:", response.json())

def test_detect(image_path):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(
            f"{BASE_URL}/api/v1/detect",
            files=files,
            data={"return_type": "json"}
        )
    print("Detection Response:", response.json())

def test_process_image(image_path):
    with open(image_path, "rb") as f:
        files = {"car_image": f}
        response = requests.post(
            f"{BASE_URL}/api/v1/process-image",
            files=files,
            data={
                "return_type": "json",
                "custom_text": "Test Image"
            }
        )
    print("Process Image Response:", response.json())

# Run tests
if __name__ == "__main__":
    test_root()
    test_ping()
    test_detect("path/to/test/image.jpg")
    test_process_image("path/to/test/image.jpg")
```

## Common Issues and Solutions

1. **File Too Large Error**
   - Maximum file size is 100MB
   - Try compressing the image or using a smaller file

2. **Invalid Image Error**
   - Make sure the file is a valid image format (JPG, PNG)
   - Check if the image is corrupted

3. **Model Loading Error**
   - Ensure `best.pt` exists in `app/models/` directory
   - Check if the model file is valid

4. **Connection Error**
   - Verify the API server is running
   - Check if the port 8000 is available

## Testing Best Practices

1. Test with different image sizes and formats
2. Test with images containing multiple license plates
3. Test with images without license plates
4. Test custom text with special characters
5. Test both JSON and image return types
6. Test error cases (invalid files, large files)
7. Test concurrent requests

## Rate Limits
- Default: 10 requests per minute per IP
- Adjust in settings if needed for testing