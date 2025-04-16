# License Plate Detection API

A FastAPI-based API for detecting and processing license plates in images.

<<<<<<< Updated upstream
## Features

- License plate detection using YOLOv5
- Image processing with custom text overlay
- File size validation (max 100MB)
- Rate limiting (10 requests per minute)
- CORS support
- Error handling and logging
- JSON and direct image responses

## Prerequisites

- Python 3.8+
- Redis server (for rate limiting)
- CUDA-compatible GPU (recommended for faster inference)

## Installation
=======
## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [API Usage](#api-usage)
6. [Troubleshooting](#troubleshooting)
7. [Development](#development)
8. [Deployment](#deployment)
>>>>>>> Stashed changes

## Prerequisites

Before you begin, ensure you have:
- Python 3.8+ installed
- Redis server installed and running
- CUDA-compatible GPU (recommended for faster inference)
- Git installed

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/SAMARTHD30/Number-Plate-Detection-API.git
cd Number-Plate-Detection-API
```

<<<<<<< Updated upstream
2. Create a virtual environment:
=======
### Step 2: Create Virtual Environment
>>>>>>> Stashed changes
```bash
# Create virtual environment
python -m venv venv
<<<<<<< Updated upstream
source venv/bin/activate  # On Windows: venv\Scripts\activate
=======

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
>>>>>>> Stashed changes
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

<<<<<<< Updated upstream
4. Set up environment variables:
Create a `.env` file in the root directory:
=======
### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory with:
>>>>>>> Stashed changes
```env
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password  # Optional
```

<<<<<<< Updated upstream
## Server Configuration

### Development Server
```bash
# Basic configuration
uvicorn app.main:app --reload

# Advanced configuration with file size handling
=======
## Configuration

### Redis Setup
```bash
# Start Redis server
# On Windows:
redis-server
# On Linux/Mac:
sudo service redis start
```

### File Size Limits
The API has a default file size limit of 100MB. To modify:
1. Update `MAX_UPLOAD_SIZE` in `app/main.py`
2. Update `MAX_UPLOAD_SIZE` in `app/api/routes.py`
3. Update Nginx configuration if using (see Deployment section)

## Running the Application

### Development Mode
```bash
# Basic configuration
uvicorn app.main:app --reload

# Advanced configuration (recommended)
>>>>>>> Stashed changes
uvicorn app.main:app --reload \
    --limit-concurrency 1 \
    --timeout-keep-alive 600 \
    --limit-max-requests 0 \
    --limit-request-line 0 \
    --limit-request-fields 0 \
    --limit-request-field-size 0
<<<<<<< Updated upstream
```

### Production Server (using Gunicorn)
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --max-requests 10000 \
    --max-requests-jitter 1000 \
    --log-level info
```

### Nginx Configuration (Production)
```nginx
server {
    listen 80;
    server_name your_domain.com;

    client_max_body_size 100M;  # Match API's file size limit
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## API Endpoints

### 1. Health Check
```http
GET /api/v1/ping
```

### 2. License Plate Detection
```http
POST /api/v1/detect
```
- Form data:
  - `image`: Image file (≤ 100MB)
  - `return_processed`: Boolean (optional) - Return processed image with bounding box

### 3. Image Processing
```http
POST /api/v1/process-image
```
- Form data:
  - `car_image`: Image file (≤ 100MB)
  - `custom_text`: String (optional) - Text to overlay on image
  - `return_type`: String (optional) - 'image' or 'json'

## File Size Limits

The API now includes robust file size validation:

1. Middleware level check (early rejection)
2. Endpoint level validation
3. Chunk-based validation for streams
4. Maximum file size: 100MB

To modify the file size limit:
1. Update `MAX_UPLOAD_SIZE` in `app/main.py`
2. Update `MAX_UPLOAD_SIZE` in `app/api/routes.py`
3. Update `client_max_body_size` in Nginx configuration

## Error Handling

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid file format)
- `413`: Payload Too Large (file > 100MB)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

### Error Response Format
```json
{
    "detail": "Error message here"
}
```

## Logging

The API uses Python's built-in logging module with DEBUG level:
- Request processing
- File size validation
- Model inference
- Error tracking

Logs are output to stdout by default.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
# Install development dependencies
pip install black isort flake8

# Format code
black .
isort .
flake8 .
```
=======
```

### Production Mode
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --max-requests 10000 \
    --max-requests-jitter 1000 \
    --log-level info
```

## API Usage

### 1. Health Check
```http
GET /api/v1/ping
```

### 2. License Plate Detection
```http
POST /api/v1/detect
```
- Form data:
  - `image`: Image file (≤ 100MB)
  - `return_processed`: Boolean (optional) - Return processed image with bounding box

### 3. Image Processing
```http
POST /api/v1/process-image
```
- Form data:
  - `car_image`: Image file (≤ 100MB)
  - `custom_text`: String (optional) - Text to overlay on image
  - `return_type`: String (optional) - 'image' or 'json'

## Troubleshooting

### Common Issues

1. **413 Request Entity Too Large**
   - Check file size (must be ≤ 100MB)
   - Verify server configuration
   - Check Nginx settings if using

2. **Rate Limiting Issues**
   - Ensure Redis server is running
   - Check Redis connection in `.env`
   - Verify rate limit settings

3. **Model Loading Issues**
   - Check CUDA availability
   - Verify model file path
   - Check file permissions

### Error Codes
- `200`: Success
- `400`: Bad Request (invalid file format)
- `413`: Payload Too Large (file > 100MB)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

## Development

### Code Style
```bash
# Install development tools
pip install black isort flake8

# Format code
black .
isort .
flake8 .
```

### Testing
```bash
# Run tests
pytest tests/
```

## Deployment

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your_domain.com;

    client_max_body_size 100M;  # Match API's file size limit
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Production Checklist
1. [ ] Configure SSL/HTTPS
2. [ ] Set up proper authentication
3. [ ] Configure monitoring and logging
4. [ ] Set up environment variables
5. [ ] Configure Nginx (if using)
6. [ ] Set up backup system
7. [ ] Configure firewall rules
>>>>>>> Stashed changes

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)
