# Additional sources for API Harvester

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

SOURCES = [
    # API Documentation Sites
    "https://docs.perplexity.ai",
    "https://docs.together.ai", 
    "https://replicate.com/docs",
    "https://docs.cohere.com",
    "https://platform.openai.com/docs",
    "https://ai.google.dev/docs",
    "https://docs.anthropic.com",
    
    # API Directories
    "https://github.com/public-apis/public-apis",
    "https://apis.guru/",
    "https://rapidapi.com/",
    
    # Trending AI Papers (for new techniques)
    "https://arxiv.org/list/cs.AI/recent",
    "https://huggingface.co/papers",
    
    # Implementation Examples
    "https://github.com/topics/api-client",
    "https://github.com/topics/ai-api"
]
