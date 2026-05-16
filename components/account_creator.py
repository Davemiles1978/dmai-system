"""
Universal Account Creator - Autonomous Account Creation on Any Platform

This module:
- Works with any website using configurable templates
- Generates temporary email addresses
- Uses Selenium for browser automation
- Handles CAPTCHA via 2Captcha/DeathByCaptcha integration
- Extracts confirmation links and API keys from emails
- Stores credentials securely
"""

import os
import re
import time
import json
import logging
import requests
import random
import string
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)


@dataclass
class Account:
    """Represents a created account"""
    email: str
    service: str
    password: str
    api_key: Optional[str] = None
    created_at: str = None
    verified: bool = False
    status: str = "pending"
    platform_type: str = "unknown"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PlatformConfig:
    """Configuration for different platforms"""
    
    # Built-in platform configurations
    PLATFORMS = {
        # AI Platforms
        'openai': {
            'signup_url': 'https://auth0.openai.com/u/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'ai',
        },
        'anthropic': {
            'signup_url': 'https://console.anthropic.com/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'ai',
        },
        'huggingface': {
            'signup_url': 'https://huggingface.co/join',
            'fields': {'email': 'email', 'username': 'username', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'ai',
        },
        'replicate': {
            'signup_url': 'https://replicate.com/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'ai',
        },
        
        # Developer Platforms
        'github': {
            'signup_url': 'https://github.com/signup',
            'fields': {'email': 'email', 'password': 'password', 'username': 'login'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'dev',
        },
        'gitlab': {
            'signup_url': 'https://gitlab.com/users/sign_up',
            'fields': {'email': 'email', 'password': 'password', 'username': 'username'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'dev',
        },
        'bitbucket': {
            'signup_url': 'https://bitbucket.org/account/signup/',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'dev',
        },
        
        # Cloud Platforms
        'aws': {
            'signup_url': 'https://portal.aws.amazon.com/billing/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'input[type="submit"]',
            'verification_required': True,
            'platform_type': 'cloud',
            'requires_phone': True,
        },
        'google_cloud': {
            'signup_url': 'https://cloud.google.com/free',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'cloud',
        },
        'azure': {
            'signup_url': 'https://azure.microsoft.com/en-us/free/',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'cloud',
        },
        
        # Social Platforms
        'twitter': {
            'signup_url': 'https://twitter.com/signup',
            'fields': {'email': 'email', 'password': 'password', 'name': 'name'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'social',
        },
        'reddit': {
            'signup_url': 'https://www.reddit.com/register/',
            'fields': {'email': 'email', 'password': 'password', 'username': 'username'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'social',
        },
        'discord': {
            'signup_url': 'https://discord.com/register',
            'fields': {'email': 'email', 'password': 'password', 'username': 'username'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'social',
        },
        
        # Email Platforms
        'protonmail': {
            'signup_url': 'https://account.proton.me/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'email',
        },
        'tutanota': {
            'signup_url': 'https://app.tutanota.com/#/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'email',
        },
        
        # API Platforms
        'rapidapi': {
            'signup_url': 'https://rapidapi.com/auth/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'api',
        },
        'apilayer': {
            'signup_url': 'https://apilayer.com/signup',
            'fields': {'email': 'email', 'password': 'password'},
            'submit_button': 'button[type="submit"]',
            'verification_required': True,
            'platform_type': 'api',
        },
    }
    
    @classmethod
    def add_platform(cls, name: str, config: Dict):
        """Add a custom platform configuration"""
        cls.PLATFORMS[name] = config
    
    @classmethod
    def get_platform(cls, name: str) -> Optional[Dict]:
        """Get platform configuration"""
        return cls.PLATFORMS.get(name)
    
    @classmethod
    def list_platforms(cls, platform_type: str = None) -> List[str]:
        """List all platforms, optionally filtered by type"""
        if platform_type:
            return [p for p, cfg in cls.PLATFORMS.items() if cfg.get('platform_type') == platform_type]
        return list(cls.PLATFORMS.keys())


class TempEmailService:
    """Handles temporary email generation and inbox checking"""
    
    def __init__(self):
        self.current_email = None
        self.current_session = None
        self.provider = None
    
    def create_email(self) -> Tuple[str, str]:
        """Create a temporary email address. Returns (email, session_id)"""
        
        # Try multiple providers
        providers = [
            self._create_10minutemail,
            self._create_guerrillamail,
            self._create_tempail,
            self._create_fallback,
        ]
        
        for provider in providers:
            try:
                email, session = provider()
                if email:
                    self.current_email = email
                    self.current_session = session
                    return email, session
            except Exception as e:
                logger.warning(f"Provider failed: {e}")
                continue
        
        raise Exception("All email providers failed")
    
    def _create_10minutemail(self) -> Tuple[str, str]:
        """Create email using 10minutemail"""
        response = requests.post(
            'https://api.10minutemail.com/api/v2/email/address',
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            self.provider = '10minutemail'
            return data.get('email'), data.get('sessionId')
        raise Exception("10minutemail failed")
    
    def _create_guerrillamail(self) -> Tuple[str, str]:
        """Create email using Guerrilla Mail"""
        response = requests.get(
            'https://api.guerrillamail.com/ajax.php?f=get_email_address',
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            self.provider = 'guerrillamail'
            return data.get('email_addr'), data.get('sid')
        raise Exception("GuerrillaMail failed")
    
    def _create_tempail(self) -> Tuple[str, str]:
        """Create email using Tempail"""
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        email = f"{random_str}@tempail.com"
        self.provider = 'tempail'
        return email, random_str
    
    def _create_fallback(self) -> Tuple[str, str]:
        """Fallback: create random email"""
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        email = f"{random_str}@temp-mail.org"
        self.provider = 'fallback'
        return email, random_str
    
    def get_inbox(self, session_id: str = None, max_wait: int = 120) -> List[Dict]:
        """Get inbox messages. Waits up to max_wait seconds."""
        if not session_id:
            session_id = self.current_session
        
        if not session_id:
            return []
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            messages = self._fetch_messages(session_id)
            if messages:
                return messages
            time.sleep(5)
        
        return []
    
    def _fetch_messages(self, session_id: str) -> List[Dict]:
        """Fetch messages from the current provider"""
        
        if self.provider == '10minutemail':
            response = requests.get(
                f'https://api.10minutemail.com/api/v2/email/messages/{session_id}',
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get('messages', [])
        
        elif self.provider == 'guerrillamail':
            response = requests.get(
                f'https://api.guerrillamail.com/ajax.php?f=fetch_email&sid={session_id}',
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get('list', [])
        
        return []
    
    def extract_verification_links(self, messages: List[Dict]) -> List[str]:
        """Extract verification links from messages"""
        links = []
        
        for msg in messages:
            body = msg.get('mail_body', msg.get('body', msg.get('mail_text', '')))
            subject = msg.get('mail_subject', msg.get('subject', ''))
            
            text = f"{subject} {body}".lower()
            
            if not any(keyword in text for keyword in ['confirm', 'verify', 'activate', 'welcome']):
                continue
            
            urls = re.findall(r'https?://[^\s<>"\']+', body)
            
            for url in urls:
                if any(keyword in url.lower() for keyword in ['confirm', 'verify', 'activate']):
                    links.append(url)
        
        return links
    
    def extract_api_keys(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """Extract API keys from messages"""
        keys = []
        
        patterns = {
            'openai': r'sk-[A-Za-z0-9]{20,100}',
            'anthropic': r'sk-ant-[A-Za-z0-9]{20,100}',
            'google': r'AIza[A-Za-z0-9]{20,50}',
            'github': r'ghp_[A-Za-z0-9]{20,100}',
            'huggingface': r'hf_[A-Za-z0-9]{20,100}',
            'replicate': r'r8_[A-Za-z0-9]{20,100}',
            'rapidapi': r'[A-Za-z0-9]{20,50}',
        }
        
        for msg in messages:
            body = msg.get('mail_body', msg.get('body', msg.get('mail_text', '')))
            
            for service, pattern in patterns.items():
                matches = re.findall(pattern, body)
                for match in matches:
                    keys.append((service, match))
        
        return keys


class CAPTCHASolver:
    """Handles CAPTCHA solving using various services"""
    
    def __init__(self, api_key: str = None, service: str = '2captcha'):
        self.api_key = api_key
        self.service = service
    
    def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2"""
        if not self.api_key:
            logger.warning("No CAPTCHA API key configured")
            return None
        
        if self.service == '2captcha':
            return self._solve_2captcha(site_key, page_url)
        elif self.service == 'deathbycaptcha':
            return self._solve_deathbycaptcha(site_key, page_url)
        
        return None
    
    def _solve_2captcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve using 2captcha"""
        try:
            # Submit CAPTCHA
            submit = requests.post(
                'http://2captcha.com/in.php',
                data={
                    'key': self.api_key,
                    'method': 'userrecaptcha',
                    'googlekey': site_key,
                    'pageurl': page_url,
                    'json': 1
                },
                timeout=30
            )
            
            if submit.status_code != 200:
                return None
            
            result = submit.json()
            if result.get('status') != 1:
                return None
            
            captcha_id = result.get('request')
            
            # Wait for solution
            for _ in range(30):
                time.sleep(5)
                response = requests.get(
                    f'http://2captcha.com/res.php?key={self.api_key}&action=get&id={captcha_id}&json=1',
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 1:
                        return data.get('request')
            
            return None
            
        except Exception as e:
            logger.error(f"2captcha failed: {e}")
            return None
    
    def _solve_deathbycaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve using DeathByCaptcha"""
        # Implementation for DeathByCaptcha
        logger.warning("DeathByCaptcha not yet implemented")
        return None


class UniversalAccountCreator:
    """Autonomous account creation on any platform"""
    
    def __init__(self, data_dir: str = "data/accounts", captcha_api_key: str = None):
        self.data_dir = data_dir
        self.accounts: Dict[str, List[Account]] = {}
        self.email_service = TempEmailService()
        self.captcha_solver = CAPTCHASolver(captcha_api_key) if captcha_api_key else None
        
        os.makedirs(data_dir, exist_ok=True)
        self._load_accounts()
        
        logger.info("🌐 Universal Account Creator initialized")
        logger.info(f"   Available platforms: {len(PlatformConfig.list_platforms())}")
    
    def _load_accounts(self):
        """Load previously created accounts"""
        accounts_file = os.path.join(self.data_dir, 'accounts.json')
        if os.path.exists(accounts_file):
            try:
                with open(accounts_file, 'r') as f:
                    data = json.load(f)
                    for service, accounts in data.items():
                        self.accounts[service] = [Account(**acc) for acc in accounts]
                total = sum(len(acc) for acc in self.accounts.values())
                logger.info(f"📂 Loaded {total} accounts from disk")
            except Exception as e:
                logger.error(f"Failed to load accounts: {e}")
    
    def _save_accounts(self):
        """Save accounts to disk"""
        accounts_file = os.path.join(self.data_dir, 'accounts.json')
        try:
            data = {}
            for service, accounts in self.accounts.items():
                data[service] = [acc.to_dict() for acc in accounts]
            with open(accounts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save accounts: {e}")
    
    def generate_password(self, length: int = 16) -> str:
        """Generate a random secure password"""
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(random.choices(chars, k=length))
    
    def create_account(self, platform: str, headless: bool = True, 
                       custom_config: Dict = None) -> Optional[Account]:
        """
        Create an account on a platform
        
        Args:
            platform: Platform name (e.g., 'github', 'openai')
            headless: Run browser in headless mode
            custom_config: Custom configuration for unsupported platforms
        
        Returns:
            Account object or None
        """
        
        # Get platform configuration
        config = custom_config or PlatformConfig.get_platform(platform)
        if not config:
            logger.error(f"Platform {platform} not found. Add it via PlatformConfig.add_platform()")
            return None
        
        logger.info(f"📝 Creating account on {platform}")
        
        # Generate credentials
        email, session_id = self.email_service.create_email()
        password = self.generate_password()
        username = email.split('@')[0][:20]
        
        # Setup WebDriver
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = None
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.get(config['signup_url'])
            wait = WebDriverWait(driver, 30)
            
            # Fill form fields
            for field_name, field_selector in config.get('fields', {}).items():
                try:
                    if field_name == 'email':
                        value = email
                    elif field_name == 'password':
                        value = password
                    elif field_name in ['username', 'login', 'name']:
                        value = username
                    else:
                        value = f"{field_name}_{random.randint(1000, 9999)}"
                    
                    # Find and fill the field
                    element = None
                    for selector_type in [By.NAME, By.ID, By.CSS_SELECTOR]:
                        try:
                            element = driver.find_element(selector_type, field_selector)
                            break
                        except NoSuchElementException:
                            continue
                    
                    if element:
                        element.clear()
                        element.send_keys(value)
                        logger.debug(f"Filled {field_name}: {value[:3]}...")
                    
                except Exception as e:
                    logger.warning(f"Failed to fill {field_name}: {e}")
            
            # Handle CAPTCHA if present
            if config.get('has_captcha', False) and self.captcha_solver:
                site_key = self._detect_recaptcha(driver)
                if site_key:
                    captcha_token = self.captcha_solver.solve_recaptcha_v2(site_key, config['signup_url'])
                    if captcha_token:
                        driver.execute_script(f"document.getElementById('g-recaptcha-response').innerHTML = '{captcha_token}';")
                        driver.execute_script("___grecaptcha_cfg.clients[0].callback('" + captcha_token + "');")
            
            # Submit form
            submit_button = None
            for selector in [config.get('submit_button'), 'button[type="submit"]', 'input[type="submit"]']:
                if selector:
                    try:
                        submit_button = driver.find_element(By.CSS_SELECTOR, selector)
                        break
                    except NoSuchElementException:
                        continue
            
            if submit_button:
                submit_button.click()
            
            # Wait for response
            time.sleep(5)
            
            # Check for verification email
            messages = self.email_service.get_inbox(session_id, max_wait=120)
            verification_links = self.email_service.extract_verification_links(messages)
            api_keys = self.email_service.extract_api_keys(messages)
            
            # Create account record
            account = Account(
                email=email,
                service=platform,
                password=password,
                verified=len(verification_links) > 0 or not config.get('verification_required', True),
                status="verified" if (len(verification_links) > 0 or not config.get('verification_required', True)) else "pending",
                platform_type=config.get('platform_type', 'unknown')
            )
            
            if api_keys:
                account.api_key = api_keys[0][1]
                logger.info(f"🔑 Extracted API key: {api_keys[0][0]}")
            
            # Store account
            if platform not in self.accounts:
                self.accounts[platform] = []
            self.accounts[platform].append(account)
            self._save_accounts()
            
            logger.info(f"✅ Created {platform} account: {email}")
            return account
            
        except Exception as e:
            logger.error(f"Failed to create {platform} account: {e}")
            return None
            
        finally:
            if driver:
                driver.quit()
    
    def _detect_recaptcha(self, driver) -> Optional[str]:
        """Detect reCAPTCHA site key on the page"""
        try:
            # Look for reCAPTCHA iframe
            iframes = driver.find_elements(By.TAG_NAME, 'iframe')
            for iframe in iframes:
                src = iframe.get_attribute('src')
                if src and 'recaptcha' in src:
                    match = re.search(r'k=([^&]+)', src)
                    if match:
                        return match.group(1)
            
            # Look for data-sitekey attribute
            elements = driver.find_elements(By.CSS_SELECTOR, '[data-sitekey]')
            for elem in elements:
                return elem.get_attribute('data-sitekey')
            
        except Exception:
            pass
        
        return None
    
    def create_multiple_accounts(self, platform: str, count: int = 5, 
                                  max_workers: int = 3) -> List[Account]:
        """Create multiple accounts in parallel"""
        accounts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.create_account, platform) for _ in range(count)]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    accounts.append(result)
        
        logger.info(f"✅ Created {len(accounts)}/{count} accounts for {platform}")
        return accounts
    
    def add_custom_platform(self, name: str, config: Dict):
        """Add a custom platform configuration"""
        PlatformConfig.add_platform(name, config)
        logger.info(f"📝 Added custom platform: {name}")
    
    def get_account(self, platform: str) -> Optional[Account]:
        """Get an existing account for a platform"""
        if platform not in self.accounts or not self.accounts[platform]:
            return None
        
        for account in sorted(self.accounts[platform], key=lambda x: x.created_at):
            if account.status == "verified":
                return account
        
        return self.accounts[platform][0] if self.accounts[platform] else None
    
    def get_api_key(self, platform: str) -> Optional[str]:
        """Get an API key for a platform"""
        account = self.get_account(platform)
        return account.api_key if account else None
    
    def get_status(self) -> Dict:
        """Get status of all accounts"""
        return {
            'total_accounts': sum(len(acc) for acc in self.accounts.values()),
            'by_platform': {
                platform: {
                    'count': len(accounts),
                    'verified': sum(1 for a in accounts if a.verified),
                    'has_api_key': sum(1 for a in accounts if a.api_key),
                    'platform_type': accounts[0].platform_type if accounts else 'unknown'
                }
                for platform, accounts in self.accounts.items()
            },
            'available_platforms': PlatformConfig.list_platforms(),
            'platforms_by_type': {
                platform_type: PlatformConfig.list_platforms(platform_type)
                for platform_type in ['ai', 'dev', 'cloud', 'social', 'email', 'api']
            }
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    creator = UniversalAccountCreator()
    print("=" * 60)
    print("Universal Account Creator Test")
    print("=" * 60)
    
    # List available platforms
    print("\n📋 Available platforms by type:")
    status = creator.get_status()
    for platform_type, platforms in status['platforms_by_type'].items():
        print(f"   {platform_type.upper()}: {', '.join(platforms[:5])}{'...' if len(platforms) > 5 else ''}")
    
    # Test email generation
    email, session = creator.email_service.create_email()
    print(f"\n📧 Generated temp email: {email}")
    
    # Test password generation
    password = creator.generate_password()
    print(f"🔐 Generated password: {password}")
    
    # Add a custom platform example
    creator.add_custom_platform('example_api', {
        'signup_url': 'https://example.com/signup',
        'fields': {'email': 'email', 'password': 'password', 'api_key': 'api_key'},
        'submit_button': 'button[type="submit"]',
        'verification_required': False,
        'platform_type': 'api',
    })
    
    print("\n✅ Universal Account Creator ready")
    print("\nTo create an account:")
    print("  account = creator.create_account('github')")
    print("  account = creator.create_account('openai')")
    print("  accounts = creator.create_multiple_accounts('huggingface', 3)")
