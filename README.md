# License Plate Detection API

A FastAPI-based API for detecting and processing license plates in images.

## Features

- License plate detection using YOLOv5 (CPU-only)
- Image processing with custom text overlay
- File size validation (max 100MB)
- Rate limiting (10 requests per minute)
- CORS support
- Error handling and logging
- JSON and direct image responses

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [API Usage](#api-usage)
6. [Troubleshooting](#troubleshooting)
7. [Development](#development)
8. [Deployment](#deployment)

## Prerequisites

Before you begin, ensure you have:
- Python 3.8+ installed
- Redis server installed and running
- Git installed

## Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/SAMARTHD30/Number-Plate-Detection-API.git
cd Number-Plate-Detection-API
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory with:
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_URL=redis://localhost:6379/0

# Model Configuration
MODEL_PATH="app/models/best.pt"

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Running the Application

### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Using gunicorn with uvicorn workers
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Usage

### Detect License Plate
```bash
# JSON response
curl -X POST "http://localhost:8000/api/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     -F "return_type=json"

# Image response
curl -X POST "http://localhost:8000/api/detect" \
     -H "accept: image/jpeg" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     -F "return_type=image" \
     --output result.jpg
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Redis Connection Issues
```bash
# Check Redis status
redis-cli ping
# Should return PONG

# Start Redis service
# Windows: Start Redis service from Services
# Linux: sudo systemctl start redis-server
# Mac: brew services start redis
```

### Model Loading Issues
- Ensure `app/models/best.pt` exists
- Check file permissions
- Verify model file integrity

### Port Conflicts
```bash
# Check port usage
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000

# Use different port if needed
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Development

### Code Structure
```
app/
├── api/
│   ├── routes.py
│   └── __init__.py
├── models/
│   └── best.pt
├── static/
├── main.py
└── __init__.py
```

### Adding New Features
1. Create new route in `app/api/routes.py`
2. Add request/response models
3. Implement business logic
4. Add error handling
5. Update documentation

## Deployment

### Server Setup
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv nginx redis-server supervisor

# Install system dependencies for OpenCV and PyTorch
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

### Application Setup
```bash
# Create directory
sudo mkdir -p /opt/license-plate-api
cd /opt/license-plate-api

# Clone repository
git clone https://github.com/SAMARTHD30/Number-Plate-Detection-API.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU only)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p app/models

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### Model Setup
```bash
# Ensure the model file is in the correct location
sudo mkdir -p /opt/license-plate-api/app/models
sudo cp path/to/your/best.pt /opt/license-plate-api/app/models/

# Set proper permissions
sudo chown -R www-data:www-data /opt/license-plate-api/app/models
sudo chmod 755 /opt/license-plate-api/app/models
```

### Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/license-plate-api

# Add this configuration:
server {
    listen 80;
    server_name your_domain.com;

    # Increase max body size for image uploads
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 75s;
        proxy_read_timeout 300s;
    }
}

# Enable the site
sudo ln -s /etc/nginx/sites-available/license-plate-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Supervisor Configuration
```bash
sudo nano /etc/supervisor/conf.d/license-plate-api.conf

# Add this configuration:
[program:license-plate-api]
command=/opt/license-plate-api/venv/bin/gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 300
directory=/opt/license-plate-api
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/license-plate-api.err.log
stdout_logfile=/var/log/license-plate-api.out.log
environment=PYTHONPATH="/opt/license-plate-api"

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start license-plate-api
```

### Redis Setup
```bash
# Start Redis server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is working
redis-cli ping
# Should return "PONG"
```

### Security Best Practices
1. Set up SSL/HTTPS using Let's Encrypt
2. Configure firewall (UFW)
3. Set proper file permissions
4. Use environment variables for sensitive data
5. Set up model access controls
6. Configure rate limiting in Redis

### Performance Optimization
1. Adjust worker count based on CPU cores
2. Configure PyTorch to use available GPU
3. Optimize model inference batch size
4. Configure proper timeouts for long-running predictions
5. Monitor memory usage and adjust as needed

### Monitoring
```bash
# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi

# Monitor CPU and Memory
htop

# Monitor API requests
tail -f /var/log/nginx/access.log

# Monitor application errors
tail -f /var/log/license-plate-api.err.log
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - See LICENSE file for details
