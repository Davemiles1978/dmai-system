#!/usr/bin/env python3
"""
DMAI API Key Validator - Hourly validation service
"""
import os
import sys
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('validator')

class KeyValidator:
    def __init__(self):
        from config import VALIDATION_URLS, DATABASE_URL
        from storage.db_manager import DatabaseManager
        from storage.encrypted_store import EncryptedKeyStore
        
        self.validation_urls = VALIDATION_URLS
        self.db = DatabaseManager(DATABASE_URL)
        
        # Get encryption key from environment
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            logger.error("ENCRYPTION_KEY not set")
            sys.exit(1)
        
        self.store = EncryptedKeyStore(encryption_key)
        
        # Load actual API keys from environment
        self.api_keys = {
            'github': os.getenv('GITHUB_TOKEN'),
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY'),
            'groq': os.getenv('GROK_API_KEY'),
            'mistral': os.getenv('MISTRAL_API_KEY'),
            'cohere': os.getenv('COHERE_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_TOKEN'),
            'deepseek': os.getenv('DEEPSEEK_API_KEY'),
        }
        
        # Log which keys are available (without exposing them)
        available_keys = [k for k, v in self.api_keys.items() if v and not v.startswith('your_')]
        logger.info(f"Available API keys in environment: {', '.join(available_keys)}")
        
    def validate_key(self, service: str, key: str) -> tuple[bool, float, str]:
        """Validate a single API key"""
        
        # Special handling for Gemini to avoid quota issues
        if service == 'gemini':
            return self._validate_gemini_key(key)
        
        # Standard validation for other services
        url = self.validation_urls.get(service)
        if not url:
            return False, 0, f"No validation URL for service: {service}"
        
        headers = self._get_headers(service, key)
        params = self._get_params(service, key)
        start = time.time()
        
        try:
            if params:
                response = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                response = requests.get(url, headers=headers, timeout=10)
                
            response_time = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return True, response_time, ""
            else:
                return False, response_time, f"HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            response_time = (time.time() - start) * 1000
            return False, response_time, "Connection error"
        except requests.exceptions.Timeout:
            response_time = (time.time() - start) * 1000
            return False, response_time, "Timeout"
        except Exception as e:
            response_time = (time.time() - start) * 1000
            return False, response_time, str(e)
    
    def _validate_gemini_key(self, key: str) -> tuple[bool, float, str]:
        """
        Specialized validation for Gemini API keys
        Uses the list models endpoint to avoid quota consumption
        """
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {'x-goog-api-key': key}
        start = time.time()
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response_time = (time.time() - start) * 1000
            
            if response.status_code == 200:
                # Success - key is valid
                return True, response_time, ""
            elif response.status_code == 403:
                # Forbidden - API not enabled or key restricted
                return False, response_time, "Forbidden - Check if Generative Language API is enabled"
            elif response.status_code == 429:
                # Quota exceeded - key is valid but busy
                logger.warning("Gemini quota exceeded, but key appears valid")
                return True, response_time, "Quota exceeded (key valid)"
            elif response.status_code == 401:
                return False, response_time, "Unauthorized - Invalid API key"
            else:
                return False, response_time, f"HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            response_time = (time.time() - start) * 1000
            return False, response_time, "Connection error"
        except requests.exceptions.Timeout:
            response_time = (time.time() - start) * 1000
            return False, response_time, "Timeout"
        except Exception as e:
            response_time = (time.time() - start) * 1000
            return False, response_time, str(e)
    
    def _get_headers(self, service: str, key: str) -> dict:
        """Get appropriate headers for each service"""
        if service == 'openai':
            return {'Authorization': f'Bearer {key}'}
        elif service == 'anthropic':
            return {'x-api-key': key, 'anthropic-version': '2023-06-01'}
        elif service == 'gemini':
            # Gemini now handled separately in _validate_gemini_key
            return {}
        elif service == 'groq':
            return {'Authorization': f'Bearer {key}'}
        elif service == 'github':
            return {'Authorization': f'token {key}', 'Accept': 'application/vnd.github.v3+json'}
        elif service == 'huggingface':
            return {'Authorization': f'Bearer {key}'}
        elif service == 'mistral':
            return {'Authorization': f'Bearer {key}'}
        elif service == 'cohere':
            return {'Authorization': f'Bearer {key}'}
        elif service == 'deepseek':
            return {'Authorization': f'Bearer {key}'}
        else:
            return {'Authorization': f'Bearer {key}'}
    
    def _get_params(self, service: str, key: str) -> dict:
        """Get URL parameters for services that need them"""
        if service == 'gemini':
            return {'key': key}
        return {}
    
    def get_key_from_environment(self, service: str) -> str:
        """Retrieve the actual API key for a service from environment"""
        service_to_env = {
            'github': 'GITHUB_TOKEN',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'groq': 'GROK_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'huggingface': 'HUGGINGFACE_TOKEN',
            'deepseek': 'DEEPSEEK_API_KEY',
        }
        
        env_var = service_to_env.get(service)
        if not env_var:
            return None
        
        return os.getenv(env_var)
    
    def run_validation_cycle(self):
        """Validate all pending keys"""
        keys = self.db.get_keys_for_validation(limit=50)
        logger.info(f"Found {len(keys)} keys pending validation")
        
        if not keys:
            logger.info("No keys to validate")
            return
        
        valid_count = 0
        for key_record in keys:
            service = key_record['service']
            key_id = key_record['id']
            
            # Get actual key from environment
            actual_key = self.get_key_from_environment(service)
            
            if not actual_key or actual_key.startswith('your_'):
                logger.warning(f"⚠️ No valid {service} key found in environment for key ID {key_id}")
                success = False
                response_time = 0
                error = f"No valid {service} key in environment"
            else:
                # Mask key for logging
                masked_key = f"{actual_key[:8]}...{actual_key[-8:]}" if len(actual_key) > 16 else "***"
                logger.info(f"🔑 Validating {service} key ID {key_id} with key: {masked_key}")
                
                # Validate the key
                success, response_time, error = self.validate_key(service, actual_key)
            
            # Update database with validation result
            self.db.update_validation(
                key_id=key_id,
                is_valid=success,
                response_time_ms=int(response_time),
                error=error
            )
            
            if success:
                valid_count += 1
                logger.info(f"✅ Key {key_id} ({service}) is VALID")
            else:
                logger.info(f"❌ Key {key_id} ({service}) is INVALID: {error}")
            
            time.sleep(1)  # Rate limiting
    
        logger.info(f"Validation complete: {valid_count}/{len(keys)} keys valid")
    
    def run(self):
        """Main loop (runs once per cron invocation)"""
        logger.info("="*60)
        logger.info("Starting validation cycle")
        logger.info("="*60)
        
        try:
            self.run_validation_cycle()
        except Exception as e:
            logger.error(f"Validation cycle failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("Validation complete")
        logger.info("="*60)

if __name__ == "__main__":
    validator = KeyValidator()
    validator.run()
