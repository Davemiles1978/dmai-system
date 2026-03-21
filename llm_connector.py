"""
DMAI LLM Connector - Provides access to multiple LLMs for intelligence
DMAI will evolve to integrate these capabilities into herself
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LLMConnector:
    """Connects to various LLM providers for intelligence"""
    
    def __init__(self):
        self.providers = {
            "openai": {
                "name": "OpenAI",
                "available": bool(os.environ.get('OPENAI_API_KEY')),
                "model": "gpt-3.5-turbo"
            },
            "google": {
                "name": "Google Gemini",
                "available": bool(os.environ.get('GOOGLE_API_KEY')),
                "model": "gemini-pro"
            },
            "deepseek": {
                "name": "DeepSeek",
                "available": bool(os.environ.get('DEEPSEEK_API_KEY')),
                "model": "deepseek-chat"
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "available": bool(os.environ.get('ANTHROPIC_API_KEY')),
                "model": "claude-3-sonnet-20240229"
            }
        }
        self._check_available()
    
    def _check_available(self):
        """Check which LLMs are available"""
        available = [p for p, info in self.providers.items() if info["available"]]
        if available:
            logger.info(f"✅ LLM providers available: {', '.join(available)}")
        else:
            logger.warning("⚠️ No LLM API keys configured. Using fallback responses.")
    
    def ask(self, prompt: str, context: str = None) -> str:
        """Ask an LLM a question"""
        # Try providers in order
        for provider in ["openai", "google", "deepseek", "anthropic"]:
            if self.providers[provider]["available"]:
                try:
                    response = self._call_provider(provider, prompt, context)
                    if response:
                        return response
                except Exception as e:
                    logger.error(f"Error calling {provider}: {e}")
                    continue
        
        # Fallback if no LLM available
        return self._fallback_response(prompt)
    
    def _call_provider(self, provider: str, prompt: str, context: str = None) -> Optional[str]:
        """Call specific LLM provider"""
        if provider == "openai":
            return self._call_openai(prompt, context)
        elif provider == "google":
            return self._call_google(prompt, context)
        elif provider == "deepseek":
            return self._call_deepseek(prompt, context)
        elif provider == "anthropic":
            return self._call_anthropic(prompt, context)
        return None
    
    def _call_openai(self, prompt: str, context: str = None) -> Optional[str]:
        """Call OpenAI API"""
        import openai
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _call_google(self, prompt: str, context: str = None) -> Optional[str]:
        """Call Google Gemini API"""
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    
    def _call_deepseek(self, prompt: str, context: str = None) -> Optional[str]:
        """Call DeepSeek API"""
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None
    
    def _call_anthropic(self, prompt: str, context: str = None) -> Optional[str]:
        """Call Anthropic Claude API"""
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback when no LLM is available"""
        return f"I'm still learning. Please configure an LLM API key (OpenAI, Google Gemini, or DeepSeek) so I can answer your questions intelligently. I'm designed to evolve and learn from these LLMs."

# Singleton instance
_instance = None

def get_llm():
    global _instance
    if _instance is None:
        _instance = LLMConnector()
    return _instance
