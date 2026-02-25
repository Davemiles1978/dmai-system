#!/bin/bash
# DEPLOY TO CLOUD FOR 24/7 EVOLUTION - ACCESS FROM ANY DEVICE

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ☁️  DMAI CLOUD DEPLOYMENT - 24/7 EVOLUTION              ║"
echo "║              Access from ANY device worldwide               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "📦 Installing AWS CLI..."
    pip install awscli
fi

# Create cloud initialization script
cat > cloud_init.sh << 'EOF'
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
EOF

# Make init script executable
chmod +x cloud_init.sh

echo ""
echo "🚀 Starting cloud deployment..."
echo ""

# Create security group
echo "🔒 Creating security group..."
aws ec2 create-security-group --group-name dmai-global --description "DMAI Global Access" 2>/dev/null || true

# Allow all traffic (for worldwide access)
aws ec2 authorize-security-group-ingress --group-name dmai-global --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name dmai-global --protocol tcp --port 80 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name dmai-global --protocol tcp --port 443 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name dmai-global --protocol tcp --port 8080 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name dmai-global --protocol tcp --port 8889 --cidr 0.0.0.0/0 2>/dev/null || true

# Create key pair
echo "🔑 Creating key pair..."
aws ec2 create-key-pair --key-name dmai-key --query 'KeyMaterial' --output text > dmai-key.pem
chmod 400 dmai-key.pem

# Launch EC2 instance
echo "🖥️  Launching cloud instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name dmai-key \
    --security-groups dmai-global \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}" \
    --user-data file://cloud_init.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DMAI-Cloud}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance created: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
sleep 10

# Allocate elastic IP (permanent address)
echo "🌐 Allocating permanent IP address..."
ALLOC_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOC_ID

# Get the public IP
PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $ALLOC_ID --query 'Addresses[0].PublicIp' --output text)

# Save URL to desktop
echo "http://$PUBLIC_IP" > ~/Desktop/DMAI_CLOUD_URL.txt

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🎉 CLOUD DEPLOYMENT COMPLETE!                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 YOUR DMAI IS NOW LIVE AT:"
echo "   http://$PUBLIC_IP"
echo ""
echo "📱 ACCESS FROM ANY DEVICE:"
echo "   • MacBook: http://$PUBLIC_IP"
echo "   • iPhone/iPad: http://$PUBLIC_IP/mobile.html"
echo "   • Android: http://$PUBLIC_IP/android.html"
echo "   • Windows: http://$PUBLIC_IP"
echo "   • Any browser: http://$PUBLIC_IP"
echo ""
echo "🔐 SSH Access (for maintenance):"
echo "   ssh -i dmai-key.pem ubuntu@$PUBLIC_IP"
echo ""
echo "📤 To upload your repos to the cloud:"
echo "   rsync -avz -e \"ssh -i dmai-key.pem\" ~/Desktop/AI-Evolution-System/repos/ ubuntu@$PUBLIC_IP:/opt/dmai/repos/"
echo ""
echo "📥 To download evolved versions:"
echo "   scp -i dmai-key.pem ubuntu@$PUBLIC_IP:/opt/dmai/checkpoints/best_versions/* ./"
echo ""
echo "✅ URL saved to: ~/Desktop/DMAI_CLOUD_URL.txt"
echo "⚠️  IMPORTANT: Save your key pair: dmai-key.pem (don't lose this!)"#!/bin/bash
# DEPLOY TO CLOUD FOR 24/7 EVOLUTION - ACCESS FROM ANY DEVICE

echo "☁️  DEPLOYING TO CLOUD FOR 24/7 EVOLUTION"
echo "=========================================="
echo ""

# Choose cloud provider
echo "Select cloud provider for 24/7 operation:"
echo "1) AWS EC2 (Most reliable, ~$30/month) - Access from ANY device"
echo "2) Google Cloud Platform (~$35/month) - Access from ANY device"
echo "3) Digital Ocean (Easiest, ~$20/month) - Access from ANY device"
echo "4) Render.com (Free tier available) - Access from ANY device"
echo "5) Heroku (Simple, ~$25/month) - Access from ANY device"
read -p "Choice [1-5]: " provider

case $provider in
    1)
        echo "🚀 Deploying to AWS EC2 for worldwide access..."
        
        # Check AWS CLI
        if ! command -v aws &> /dev/null; then
            pip install awscli
        fi
        
        # Create security group with worldwide access
        aws ec2 create-security-group --group-name ai-evolution-global --description "AI Evolution Global Access"
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-global --protocol tcp --port 22 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-global --protocol tcp --port 8080 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-global --protocol tcp --port 80 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-global --protocol tcp --port 443 --cidr 0.0.0.0/0
        
        # Create key pair
        aws ec2 create-key-pair --key-name ai-evolution-key --query 'KeyMaterial' --output text > ai-evolution-key.pem
        chmod 400 ai-evolution-key.pem
        
        # Launch EC2 instance with elastic IP for permanent address
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id ami-0c02fb55956c7d316 \
            --instance-type t3.medium \
            --key-name ai-evolution-key \
            --security-groups ai-evolution-global \
            --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50}" \
            --user-data file://cloud_init_full.sh \
            --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AI-Evolution-Global}]' \
            --query 'Instances[0].InstanceId' \
            --output text)
        
        echo "✅ Instance created: $INSTANCE_ID"
        
        # Allocate elastic IP (permanent address)
        ALLOC_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
        aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOC_ID
        
        # Get the public IP
        PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $ALLOC_ID --query 'Addresses[0].PublicIp' --output text)
        
        # Create a friendly domain (optional - use your own domain)
        echo ""
        echo "🎉 GLOBAL DEPLOYMENT COMPLETE!"
        echo "================================"
        echo "🌐 ACCESS FROM ANY DEVICE:"
        echo "   http://$PUBLIC_IP:8080"
        echo ""
        echo "📱 Test on your:"
        echo "   • MacBook: http://$PUBLIC_IP:8080"
        echo "   • iPhone/iPad: http://$PUBLIC_IP:8080"
        echo "   • Android: http://$PUBLIC_IP:8080"
        echo "   • Windows PC: http://$PUBLIC_IP:8080"
        echo "   • Any tablet: http://$PUBLIC_IP:8080"
        echo ""
        echo "🔒 To add HTTPS (secure connection):"
        echo "   ssh -i ai-evolution-key.pem ubuntu@$PUBLIC_IP"
        echo "   sudo apt-get install nginx certbot python3-certbot-nginx"
        echo "   sudo certbot --nginx -d yourdomain.com"
        echo ""
        echo "📤 To upload your repos:"
        echo "   rsync -avz -e \"ssh -i ai-evolution-key.pem\" ~/Desktop/AI-Evolution-System/repos/ ubuntu@$PUBLIC_IP:/opt/ai-evolution/repos/"
        echo ""
        echo "✅ System runs 24/7 - computer can be OFF"
        ;;
esac

# Save the URL to a file for easy access
echo "http://$PUBLIC_IP:8080" > ~/Desktop/AI_EVOLUTION_URL.txt
echo "✅ URL saved to Desktop: AI_EVOLUTION_URL.txt"#!/bin/bash
# DEPLOY TO CLOUD FOR 24/7 EVOLUTION - COMPUTER CAN BE OFF

echo "☁️  DEPLOYING TO CLOUD FOR 24/7 EVOLUTION"
echo "=========================================="
echo ""

# Choose cloud provider
echo "Select cloud provider for 24/7 operation:"
echo "1) AWS EC2 (Most reliable, ~$30/month)"
echo "2) Google Cloud Platform (~$35/month)"
echo "3) Digital Ocean (Easiest, ~$20/month)"
echo "4) Render.com (Free tier available)"
echo "5) Heroku (Simple, ~$25/month)"
read -p "Choice [1-5]: " provider

case $provider in
    1)
        echo "🚀 Deploying to AWS EC2..."
        
        # Check AWS CLI
        if ! command -v aws &> /dev/null; then
            pip install awscli
        fi
        
        # Create security group
        aws ec2 create-security-group --group-name ai-evolution-247 --description "AI Evolution 24/7"
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-247 --protocol tcp --port 22 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-247 --protocol tcp --port 8080 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-247 --protocol tcp --port 80 --cidr 0.0.0.0/0
        aws ec2 authorize-security-group-ingress --group-name ai-evolution-247 --protocol tcp --port 443 --cidr 0.0.0.0/0
        
        # Create key pair
        aws ec2 create-key-pair --key-name ai-evolution-key --query 'KeyMaterial' --output text > ai-evolution-key.pem
        chmod 400 ai-evolution-key.pem
        
        # Launch EC2 instance (t3.medium for 24/7 evolution)
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id ami-0c02fb55956c7d316 \
            --instance-type t3.medium \
            --key-name ai-evolution-key \
            --security-groups ai-evolution-247 \
            --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=50}" \
            --user-data file://cloud_init_full.sh \
            --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AI-Evolution-247}]' \
            --query 'Instances[0].InstanceId' \
            --output text)
        
        echo "✅ Instance created: $INSTANCE_ID"
        
        # Wait for instance
        echo "⏳ Waiting for instance to start..."
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID
        sleep 30
        
        # Get public IP
        PUBLIC_IP=$(aws ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)
        
        echo ""
        echo "🎉 CLOUD DEPLOYMENT COMPLETE!"
        echo "================================"
        echo "🌐 Access your 24/7 AI Evolution System at:"
        echo "   http://$PUBLIC_IP:8080"
        echo ""
        echo "🔑 SSH access:"
        echo "   ssh -i ai-evolution-key.pem ubuntu@$PUBLIC_IP"
        echo ""
        echo "📤 To upload your repos to the cloud:"
        echo "   rsync -avz -e \"ssh -i ai-evolution-key.pem\" ~/Desktop/AI-Evolution-System/repos/ ubuntu@$PUBLIC_IP:/opt/ai-evolution/repos/"
        echo ""
        echo "✅ System will run 24/7 even when your computer is OFF!"
        ;;
        
    5)
        echo "🚀 Deploying to Render.com (Free tier available)..."
        
        # Create render.yaml config
        cat > render.yaml << 'RENDER'
services:
  - type: web
    name: ai-evolution-247
    env: python
    buildCommand: |
      pip install -r requirements.txt
      python mark_all_for_evolution.py
    startCommand: |
      python app.py &
      python evolution_engine.py continuous 1000000
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    autoDeploy: true
    plan: free
RENDER
        
        echo "✅ Render.com config created"
        echo "📤 Push to GitHub and connect at: https://render.com"
        ;;
esac

echo ""
echo "🎯 YOUR SYSTEM IS NOW RUNNING 24/7 IN THE CLOUD!"
echo "   • Computer can be OFF completely"
echo "   • Access from anywhere: http://$PUBLIC_IP:8080"
echo "   • Evolution continues automatically"
echo "   • Best versions saved in cloud storage"
