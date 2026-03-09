"""Handle notifications to different devices"""

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationService:
    """Send notifications to various endpoints"""
    
    def __init__(self, config_file='device_registry/notify_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load notification config"""
        if os.path.exists(self.config_file):
            import json
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Empty config - user will set up later
            return {"email": {}, "pushover": {}, "telegram": {}}
    
    def save_config(self):
        """Save notification config"""
        import json
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def send_email(self, to_email, subject, body):
        """Send email notification"""
        # Check if email configured
        email_config = self.config.get('email', {})
        if not email_config.get('enabled'):
            logger.info("Email not configured. Would send:")
            logger.info(f"To: {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body: {body}")
            return False
        
        # Actual email sending (when configured)
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False
    
    def send_push(self, title, message, device=None):
        """Send push notification (placeholder - implement with Pushover/APNS)"""
        logger.info(f"Push notification: {title} - {message}")
        
        # Simulate for now
        print(f"\n📱 [PUSH] {title}: {message}")
        return True
    
    def configure_email(self, from_addr, password, smtp_server="smtp.gmail.com", smtp_port=587):
        """Configure email settings"""
        self.config['email'] = {
            'enabled': True,
            'from': from_addr,
            'password': password,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port
        }
        self.save_config()
        logger.info("Email configured")

# Test notifications
if __name__ == "__main__":
    ns = NotificationService()
    
    print("Testing notifications:")
    ns.send_push("DMAI Update", "Your video is ready!")
    ns.send_email("david@example.com", "DMAI Result", "Your video is ready at https://dmai.io/vid")
