# Deployment Guide for License Plate API

## 1. Server Setup

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv nginx redis-server supervisor

# Install system dependencies for OpenCV and PyTorch
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

## 2. Application Setup

```bash
# Create directory
sudo mkdir -p /opt/license-plate-api
cd /opt/license-plate-api

# Clone repository
git clone https://github.com/SAMARTHD30/Number-Plate-Detection-API.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch first (choose one based on your setup)
# For CUDA (if you have NVIDIA GPU):
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
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

## 3. Model Setup

```bash
# Ensure the model file is in the correct location
# Place your YOLOv5 model file (best.pt) in the app/models directory
sudo mkdir -p /opt/license-plate-api/app/models
sudo cp path/to/your/best.pt /opt/license-plate-api/app/models/

# Set proper permissions
sudo chown -R www-data:www-data /opt/license-plate-api/app/models
sudo chmod 755 /opt/license-plate-api/app/models
```

## 4. Nginx Configuration

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

## 5. Supervisor Configuration

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

## 6. Redis Setup

```bash
# Start Redis server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is working
redis-cli ping
# Should return "PONG"
```

## 7. Testing

```bash
# Test API status
curl http://localhost:8000/api/v1/ping

# Test model endpoint (replace with actual image path)
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/test/image.jpg" \
  -F "return_processed=true"

# Check logs
sudo tail -f /var/log/license-plate-api.err.log
sudo tail -f /var/log/nginx/error.log
```

## 8. Common Commands

```bash
# Restart application
sudo supervisorctl restart license-plate-api

# View logs
sudo tail -f /var/log/license-plate-api.err.log

# Check Redis status
sudo systemctl status redis-server

# Monitor GPU usage (if using CUDA)
nvidia-smi

# Check model predictions
curl -X POST http://localhost:8000/api/v1/detect \
  -F "image=@test.jpg" \
  -F "return_processed=true" \
  --output result.jpg
```

## 9. Security Notes

1. Set up SSL/HTTPS using Let's Encrypt
2. Configure firewall (UFW)
3. Set proper file permissions
4. Use environment variables for sensitive data
5. Set up model access controls
6. Configure rate limiting in Redis

## 10. Performance Optimization

1. Adjust worker count based on CPU cores
2. Configure PyTorch to use available GPU
3. Optimize model inference batch size
4. Configure proper timeouts for long-running predictions
5. Monitor memory usage and adjust as needed

## 11. Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and Memory
htop

# Monitor API requests
tail -f /var/log/nginx/access.log

# Monitor application errors
tail -f /var/log/license-plate-api.err.log
```
