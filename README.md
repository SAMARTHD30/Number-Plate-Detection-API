# License Plate Detection API

A FastAPI-based API for license plate detection with rate limiting and image processing capabilities.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/SAMARTHD30/Number-Plate-Detection-API.git
cd Number-Plate-Detection-API
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Redis server (required for rate limiting)

5. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Check API Status
```bash
curl -X GET http://localhost:8000/api/v1/ping
```

### Detect License Plate
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:8000/api/v1/detect
```

### Process Image
```bash
curl -X POST \
  -F "car_image=@path/to/car.jpg" \
  -F "custom_text=ABC123" \
  http://localhost:8000/api/v1/process
```

### Get Processed Image
```bash
curl -X GET http://localhost:8000/api/v1/image/{image_id}
```

## Deployment

See `deployment.md` for detailed deployment instructions.

## Rate Limiting

The API includes rate limiting (10 requests per minute) using Redis.

## Dependencies

- FastAPI
- Uvicorn
- OpenCV (Headless)
- Redis
- FastAPI Limiter
- Python Multipart

## Development

1. Install development dependencies:
```bash
pip install pytest black isort
```

2. Format code:
```bash
black .
isort .
```

## Production Notes

1. Use proper SSL/HTTPS in production
2. Configure proper authentication
3. Set up monitoring and logging
4. Use environment variables for configuration
