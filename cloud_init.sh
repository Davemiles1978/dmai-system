#!/bin/bash
# CLOUD INIT - Runs when VM starts

echo "🚀 Initializing DMAI Cloud Instance..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Python and dependencies
apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# Create app directory
mkdir -p /opt/dmai
cd /opt/dmai

# Copy your AI Evolution System (this will be done via rsync)
# For now, create basic structure
mkdir -p {repos,checkpoints,logs,ui}

# Create systemd service for 24/7 evolution
cat > /etc/systemd/system/dmai-evolution.service << 'SERVICE'
[Unit]
Description=DMAI Evolution Engine
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dmai
ExecStart=/usr/bin/python3 /opt/dmai/evolution_engine.py continuous 1000000
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
SERVICE

# Create systemd service for web UI
cat > /etc/systemd/system/dmai-web.service << 'WEB'
[Unit]
Description=DMAI Web UI
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dmai
ExecStart=/usr/bin/python3 -m http.server 8080 --bind 0.0.0.0 --directory ui
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
WEB

# Create API service
cat > /etc/systemd/system/dmai-api.service << 'API'
[Unit]
Description=DMAI API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dmai
ExecStart=/usr/bin/python3 /opt/dmai/api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
API

# Enable and start services
systemctl enable dmai-evolution.service
systemctl enable dmai-web.service
systemctl enable dmai-api.service
systemctl start dmai-evolution.service
systemctl start dmai-web.service
systemctl start dmai-api.service

# Setup Nginx as reverse proxy (for HTTPS)
cat > /etc/nginx/sites-available/dmai << 'NGINX'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api {
        proxy_pass http://localhost:8889;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
NGINX

ln -s /etc/nginx/sites-available/dmai /etc/nginx/sites-enabled/
systemctl restart nginx

# Setup daily backup cron
cat > /etc/cron.daily/dmai-backup << 'BACKUP'
#!/bin/bash
cd /opt/dmai
tar -czf /root/dmai-backup-$(date +%Y%m%d).tar.gz repos/ checkpoints/ logs/
aws s3 cp /root/dmai-backup-*.tar.gz s3://dmai-backups/ 2>/dev/null || true
BACKUP

chmod +x /etc/cron.daily/dmai-backup

echo "✅ DMAI Cloud initialization complete!"
echo "🌐 Web UI: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
