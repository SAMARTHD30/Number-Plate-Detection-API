# Deployment Guide for License Plate API

## 1. Server Setup

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv nginx redis-server supervisor
```

## 2. Application Setup

```bash
# Create directory
sudo mkdir -p /opt/license-plate-api
cd /opt/license-plate-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/license-plate-api

# Add this configuration:
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable the site
sudo ln -s /etc/nginx/sites-available/license-plate-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 4. Supervisor Configuration

```bash
sudo nano /etc/supervisor/conf.d/license-plate-api.conf

# Add this configuration:
[program:license-plate-api]
command=/opt/license-plate-api/venv/bin/gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
directory=/opt/license-plate-api
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/license-plate-api.err.log
stdout_logfile=/var/log/license-plate-api.out.log

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start license-plate-api
```

## 5. Redis Setup

```bash
# Start Redis server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## 6. Testing

```bash
# Test API status
curl http://localhost:8000/api/v1/ping

# Check logs
sudo tail -f /var/log/license-plate-api.err.log
sudo tail -f /var/log/nginx/error.log
```

## 7. Common Commands

```bash
# Restart application
sudo supervisorctl restart license-plate-api

# View logs
sudo tail -f /var/log/license-plate-api.err.log

# Check Redis status
sudo systemctl status redis-server
```

## 8. Security Notes

1. Set up SSL/HTTPS using Let's Encrypt
2. Configure firewall (UFW)
3. Set proper file permissions
4. Use environment variables for sensitive data