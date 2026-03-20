#!/usr/bin/env python3
"""
GitHub Scraper Evolution - Working version with None token handling
"""
import logging
logger = logging.getLogger("github_scraper_evolution")

class GitHubScraperEvolution:
    def __init__(self, config):
        self.config = config
        self.token = config.get('github_token')
        if self.token:
            logger.info(f"✅ GitHubScraperEvolution initialized with token: {self.token[:10]}...")
        else:
            logger.warning("⚠️ No GitHub token provided - running in public mode (rate limited)")
    
    def search_github(self, query):
        logger.info(f"Searching GitHub for: {query}")
        return []
