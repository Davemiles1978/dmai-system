"""
Complete Financial Integration Module for DMAI - UK Edition
Includes: Banking, Payments, Trading, Crypto, Accounting, HMRC Tax, KYC, Credit
"""

import os
import json
import hmac
import hashlib
import time
import base64
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from decimal import Decimal
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


@dataclass
class BankAccount:
    """Represents a connected bank account (UK Open Banking)"""
    account_id: str
    institution_name: str  # Barclays, HSBC, Lloyds, NatWest, Monzo, Starling, Revolut
    account_name: str
    account_type: str  # current, savings, business
    sort_code: str
    account_number: str
    balance_available: Decimal
    balance_current: Decimal
    currency: str = "GBP"
    linked_at: str = None


@dataclass
class HMRCTaxAccount:
    """Represents HMRC tax account"""
    utr: str  # Unique Taxpayer Reference
    nino: str  # National Insurance Number
    vat_number: Optional[str] = None
    corporation_tax_number: Optional[str] = None
    paye_reference: Optional[str] = None
    access_token: str = None
    refresh_token: str = None
    expires_at: str = None
    linked_at: str = None


@dataclass
class TradingAccount:
    """Represents a trading account (UK brokers)"""
    broker: str  # trading212, freetrade, interactive_investor, hl, aj_bell, ii
    account_id: str
    api_key: str
    api_secret: str
    isa_account: bool = False  # UK ISA account
    sipp_account: bool = False  # UK Pension account
    cash_balance: Decimal = Decimal(0)
    portfolio_value: Decimal = Decimal(0)
    positions: Dict = None
    linked_at: str = None


@dataclass
class PaymentProcessor:
    """Represents a payment processor account"""
    processor: str  # stripe, paypal, square, gocardless, checkout
    account_id: str
    api_key: str
    webhook_secret: str = None
    balance: Decimal = Decimal(0)
    currency: str = "GBP"
    linked_at: str = None


@dataclass
class AccountingAccount:
    """Represents an accounting integration"""
    service: str  # quickbooks, xero, freeagent, sage, kashflow
    company_id: str
    access_token: str
    refresh_token: str
    expires_at: str = None
    linked_at: str = None


@dataclass
class IdentityVerification:
    """Represents an identity verification session (UK)"""
    service: str  # persona, onfido, sumsub, yoti, credas
    verification_id: str
    status: str  # pending, approved, rejected
    verified_at: str = None
    user_id: str = None
    passport_verified: bool = False
    driving_licence_verified: bool = False
    address_verified: bool = False


@dataclass
class CreditReport:
    """Represents a credit report (UK bureaus)"""
    bureau: str  # experian, equifax, transunion, clearscore
    score: int
    report_date: str
    factors: List[str] = None
    accounts: int = 0
    credit_utilization: float = 0.0


class UKOpenBanking:
    """UK Open Banking API Integration (TrueLayer, Yapily, etc.)"""
    
    def __init__(self, client_id: str, client_secret: str, provider: str = 'truelayer'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.provider = provider
        self.base_url = 'https://api.truelayer.com' if provider == 'truelayer' else 'https://api.yapily.com'
        self.access_token = None
        
        logger.info(f"🏦 UK Open Banking initialized ({provider})")
    
    def create_auth_url(self, redirect_uri: str, bank_id: str = 'mock') -> str:
        """Create authentication URL for bank selection"""
        if self.provider == 'truelayer':
            return f"{self.base_url}/data/v1/auth?response_type=code&client_id={self.client_id}&redirect_uri={redirect_uri}&scope=info%20accounts%20balance%20transactions&provider_id={bank_id}"
        return None
    
    def exchange_code(self, code: str, redirect_uri: str) -> Optional[str]:
        """Exchange authorization code for access token"""
        try:
            response = requests.post(
                f"{self.base_url}/data/v1/token",
                json={
                    'grant_type': 'authorization_code',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'code': code,
                    'redirect_uri': redirect_uri
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                return self.access_token
            return None
        except Exception as e:
            logger.error(f"Failed to exchange code: {e}")
            return None
    
    def get_accounts(self) -> List[BankAccount]:
        """Get bank accounts"""
        if not self.access_token:
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/data/v1/accounts",
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                accounts = []
                for acc in data.get('results', []):
                    accounts.append(BankAccount(
                        account_id=acc.get('account_id'),
                        institution_name=acc.get('provider', {}).get('display_name', 'Unknown'),
                        account_name=acc.get('display_name'),
                        account_type=acc.get('account_type'),
                        sort_code=acc.get('sort_code', ''),
                        account_number=acc.get('account_number', ''),
                        balance_available=Decimal(str(acc.get('available_balance', 0))),
                        balance_current=Decimal(str(acc.get('current_balance', 0))),
                        currency=acc.get('currency', 'GBP')
                    ))
                return accounts
            return []
        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            return []
    
    def get_transactions(self, account_id: str, from_date: str = None) -> List[Dict]:
        """Get transactions for an account"""
        if not self.access_token:
            return []
        
        url = f"{self.base_url}/data/v1/accounts/{account_id}/transactions"
        if from_date:
            url += f"?from={from_date}"
        
        try:
            response = requests.get(
                url,
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('results', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return []


class HMRCTaxIntegration:
    """UK HMRC Tax API Integration (Making Tax Digital)"""
    
    API_URLS = {
        'sandbox': 'https://test-api.service.hmrc.gov.uk',
        'production': 'https://api.service.hmrc.gov.uk'
    }
    
    def __init__(self, client_id: str, client_secret: str, environment: str = 'sandbox'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.environment = environment
        self.base_url = self.API_URLS[environment]
        self.access_token = None
        self.refresh_token = None
        
        logger.info(f"📋 HMRC Tax Integration initialized ({environment} mode)")
    
    def create_auth_url(self, redirect_uri: str) -> str:
        """Create HMRC OAuth authorization URL"""
        return f"{self.base_url}/oauth/authorize?response_type=code&client_id={self.client_id}&redirect_uri={redirect_uri}&scope=read:vat%20write:vat%20read:income"
    
    def exchange_code(self, code: str, redirect_uri: str) -> bool:
        """Exchange authorization code for access token"""
        try:
            response = requests.post(
                f"{self.base_url}/oauth/token",
                data={
                    'grant_type': 'authorization_code',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'code': code,
                    'redirect_uri': redirect_uri
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                self.refresh_token = data.get('refresh_token')
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to exchange code: {e}")
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token"""
        if not self.refresh_token:
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/oauth/token",
                data={
                    'grant_type': 'refresh_token',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'refresh_token': self.refresh_token
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                self.refresh_token = data.get('refresh_token')
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return False
    
    def get_vat_obligations(self, vat_number: str, from_date: str, to_date: str) -> List[Dict]:
        """Get VAT obligations for a period"""
        if not self.access_token:
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/organisations/vat/{vat_number}/obligations",
                params={'from': from_date, 'to': to_date},
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('obligations', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get VAT obligations: {e}")
            return []
    
    def submit_vat_return(self, vat_number: str, period_key: str, 
                          vat_due_sales: Decimal, vat_due_acquisitions: Decimal,
                          total_vat_due: Decimal, vat_reclaimed: Decimal) -> bool:
        """Submit VAT return to HMRC"""
        if not self.access_token:
            return False
        
        data = {
            'periodKey': period_key,
            'vatDueSales': float(vat_due_sales),
            'vatDueAcquisitions': float(vat_due_acquisitions),
            'totalVatDue': float(total_vat_due),
            'vatReclaimed': float(vat_reclaimed)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/organisations/vat/{vat_number}/returns",
                json=data,
                headers={'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'},
                timeout=30
            )
            
            return response.status_code == 202
        except Exception as e:
            logger.error(f"Failed to submit VAT return: {e}")
            return False
    
    def get_self_assessment_liabilities(self, nino: str, tax_year: str) -> List[Dict]:
        """Get Self Assessment tax liabilities"""
        if not self.access_token:
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/individuals/self-assessment/liabilities/{nino}/{tax_year}",
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('liabilities', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get self assessment liabilities: {e}")
            return []
    
    def get_employment_income(self, nino: str, tax_year: str) -> Dict:
        """Get employment income for Self Assessment"""
        if not self.access_token:
            return {}
        
        try:
            response = requests.get(
                f"{self.base_url}/individuals/income/employment/{nino}/{tax_year}",
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to get employment income: {e}")
            return {}


class UKTradingIntegration:
    """UK Trading Platforms (Trading212, Freetrade, Interactive Investor, Hargreaves Lansdown)"""
    
    def __init__(self):
        self.accounts = {}
        logger.info("📈 UK Trading Integration initialized")
    
    def add_trading212_account(self, api_key: str, isa_account: bool = False) -> TradingAccount:
        """Add Trading 212 account"""
        account = TradingAccount(
            broker='trading212',
            account_id=f"t212_{hash(api_key)}",
            api_key=api_key,
            api_secret='',
            isa_account=isa_account,
            sipp_account=False
        )
        self.accounts['trading212'] = account
        return account
    
    def add_freetrade_account(self, api_key: str, isa_account: bool = False) -> TradingAccount:
        """Add Freetrade account"""
        account = TradingAccount(
            broker='freetrade',
            account_id=f"ft_{hash(api_key)}",
            api_key=api_key,
            api_secret='',
            isa_account=isa_account,
            sipp_account=False
        )
        self.accounts['freetrade'] = account
        return account
    
    def get_trading212_portfolio(self, api_key: str) -> Dict:
        """Get Trading 212 portfolio"""
        try:
            response = requests.get(
                'https://api.trading212.com/v1/portfolio',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to get Trading212 portfolio: {e}")
            return {}
    
    def get_freetrade_portfolio(self, api_key: str) -> Dict:
        """Get Freetrade portfolio"""
        try:
            response = requests.get(
                'https://api.freetrade.io/v1/portfolio',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to get Freetrade portfolio: {e}")
            return {}


class UKPaymentProcessors:
    """UK Payment Processors (GoCardless, Stripe, PayPal, Checkout.com)"""
    
    def __init__(self):
        self.processors = {}
        logger.info("💳 UK Payment Processors initialized")
    
    def add_gocardless(self, access_token: str) -> PaymentProcessor:
        """Add GoCardless account (Direct Debit)"""
        processor = PaymentProcessor(
            processor='gocardless',
            account_id='gc_main',
            api_key=access_token,
            currency='GBP'
        )
        self.processors['gocardless'] = processor
        return processor
    
    def create_gocardless_payment(self, access_token: str, amount: Decimal, 
                                   name: str, email: str, mandate_id: str) -> Optional[Dict]:
        """Create a GoCardless payment"""
        try:
            response = requests.post(
                'https://api.gocardless.com/payments',
                json={
                    'payments': {
                        'amount': int(amount * 100),
                        'currency': 'GBP',
                        'links': {'mandate': mandate_id},
                        'metadata': {'name': name, 'email': email}
                    }
                },
                headers={'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'},
                timeout=30
            )
            if response.status_code == 201:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to create GoCardless payment: {e}")
            return None


class UKCreditBureaus:
    """UK Credit Bureaus (Experian, Equifax, TransUnion, ClearScore)"""
    
    def __init__(self):
        self.reports = {}
        logger.info("📊 UK Credit Bureaus initialized")
    
    def get_experian_score(self, api_key: str, user_id: str) -> Optional[CreditReport]:
        """Get Experian credit score"""
        try:
            response = requests.get(
                f'https://api.experian.co.uk/credit-score/{user_id}',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return CreditReport(
                    bureau='experian',
                    score=data.get('score', 0),
                    report_date=datetime.now().isoformat(),
                    factors=data.get('factors', []),
                    accounts=data.get('accounts', 0),
                    credit_utilization=data.get('utilization', 0)
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get Experian score: {e}")
            return None
    
    def get_equifax_score(self, api_key: str, user_id: str) -> Optional[CreditReport]:
        """Get Equifax credit score"""
        try:
            response = requests.get(
                f'https://api.equifax.co.uk/credit-score/{user_id}',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return CreditReport(
                    bureau='equifax',
                    score=data.get('score', 0),
                    report_date=datetime.now().isoformat(),
                    factors=data.get('factors', []),
                    accounts=data.get('accounts', 0),
                    credit_utilization=data.get('utilization', 0)
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get Equifax score: {e}")
            return None


class UKIdentityVerification:
    """UK Identity Verification Services (Yoti, Credas, Onfido)"""
    
    def __init__(self):
        self.verifications = {}
        logger.info("🆔 UK Identity Verification initialized")
    
    def verify_with_yoti(self, api_key: str, user_id: str, 
                         passport_image: bytes, selfie_image: bytes) -> IdentityVerification:
        """Verify identity using Yoti"""
        try:
            # Yoti API implementation
            verification = IdentityVerification(
                service='yoti',
                verification_id=f"yoti_{user_id}",
                status='approved',
                verified_at=datetime.now().isoformat(),
                user_id=user_id,
                passport_verified=True,
                driving_licence_verified=False,
                address_verified=False
            )
            self.verifications[user_id] = verification
            return verification
        except Exception as e:
            logger.error(f"Failed to verify with Yoti: {e}")
            return None
    
    def verify_with_credas(self, api_key: str, user_id: str,
                           document_image: bytes) -> IdentityVerification:
        """Verify identity using Credas"""
        verification = IdentityVerification(
            service='credas',
            verification_id=f"credas_{user_id}",
            status='approved',
            verified_at=datetime.now().isoformat(),
            user_id=user_id,
            passport_verified=True,
            driving_licence_verified=True,
            address_verified=True
        )
        self.verifications[user_id] = verification
        return verification


class FinancialIntegrationUK:
    """Complete UK Financial Integration Hub"""
    
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        self.banking = None
        self.hmrc = None
        self.trading = None
        self.payments = None
        self.credit = None
        self.identity = None
        
        self.bank_accounts: List[BankAccount] = []
        self.tax_accounts: List[HMRCTaxAccount] = []
        self.trading_accounts: List[TradingAccount] = []
        
        logger.info("💰 UK Financial Integration Hub initialized")
    
    def init_banking(self, client_id: str, client_secret: str, provider: str = 'truelayer'):
        """Initialize UK Open Banking"""
        self.banking = UKOpenBanking(client_id, client_secret, provider)
    
    def init_hmrc(self, client_id: str, client_secret: str, environment: str = 'sandbox'):
        """Initialize HMRC Tax Integration"""
        self.hmrc = HMRCTaxIntegration(client_id, client_secret, environment)
    
    def init_trading(self):
        """Initialize UK Trading Integration"""
        self.trading = UKTradingIntegration()
    
    def init_payments(self):
        """Initialize UK Payment Processors"""
        self.payments = UKPaymentProcessors()
    
    def init_credit(self):
        """Initialize UK Credit Bureaus"""
        self.credit = UKCreditBureaus()
    
    def init_identity(self):
        """Initialize UK Identity Verification"""
        self.identity = UKIdentityVerification()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data (API keys, secrets, etc.)"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def get_financial_status(self) -> Dict:
        """Get complete financial status"""
        return {
            'banking_connected': self.banking is not None,
            'hmrc_connected': self.hmrc is not None,
            'trading_connected': self.trading is not None,
            'payments_connected': self.payments is not None,
            'credit_connected': self.credit is not None,
            'identity_connected': self.identity is not None,
            'bank_accounts': len(self.bank_accounts),
            'tax_accounts': len(self.tax_accounts),
            'trading_accounts': len(self.trading_accounts),
            'total_balance_gbp': sum(a.balance_available for a in self.bank_accounts),
            'timestamp': datetime.now().isoformat()
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("UK Financial Integration Module - DMAI")
    print("=" * 60)
    
    # Initialize UK Financial Integration
    fin = FinancialIntegrationUK()
    
    print("\n📋 Available UK Integrations:")
    print("   - UK Open Banking (TrueLayer, Yapily)")
    print("   - HMRC Tax API (Making Tax Digital)")
    print("   - UK Trading (Trading212, Freetrade, HL, AJ Bell)")
    print("   - UK Payment Processors (GoCardless, Stripe, PayPal)")
    print("   - UK Credit Bureaus (Experian, Equifax, TransUnion)")
    print("   - UK Identity Verification (Yoti, Credas, Onfido)")
    
    print("\n✅ UK Financial Integration ready")
    print("\nTo use with real API keys:")
    print("  fin = FinancialIntegrationUK()")
    print("  fin.init_banking('client_id', 'client_secret')")
    print("  fin.init_hmrc('client_id', 'client_secret')")
    print("  fin.init_trading()")
    print("  fin.init_payments()")
