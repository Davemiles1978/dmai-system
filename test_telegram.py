import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

print(f"Token from env: {token[:10]}... (hidden)")
print(f"Chat ID: {chat_id}")

# Test the token
response = requests.get(f"https://api.telegram.org/bot{token}/getMe")
print(f"Bot info: {response.json()}")

# Send a test message
url = f"https://api.telegram.org/bot{token}/sendMessage"
data = {'chat_id': chat_id, 'text': '🧪 Direct test from DMAI'}
response = requests.post(url, data=data)
print(f"Send result: {response.json()}")
