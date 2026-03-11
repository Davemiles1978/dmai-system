#!/usr/bin/env python3
"""Book Reader Service - DMAI"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import logging
import requests
from datetime import datetime
from core.paths import VOCAB_PATH, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("BOOK")

class BookReader:
    def __init__(self):
        self.vocab_path = VOCAB_PATH
        self.data_dir = DATA_DIR
        self.books_read = 0
        logger.info(f"📚 Vocabulary path: {self.vocab_path}")
        logger.info(f"📂 Data directory: {self.data_dir}")
        
    def load_vocabulary(self):
        """Load existing vocabulary"""
        try:
            if self.vocab_path.exists():
                with open(self.vocab_path, 'r') as f:
                    vocab = json.load(f)
                logger.info(f"📚 Loaded {len(vocab)} words")
                return vocab
            else:
                logger.warning(f"Vocabulary not found at {self.vocab_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return {}
    
    def download_book(self, url):
        """Download a book from Project Gutenberg with retry logic"""
        for retry in range(3):
            try:
                logger.info(f"Downloading book from {url} (attempt {retry+1}/3)")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                logger.info(f"✅ Successfully downloaded book from {url}")
                return response.text
            except requests.exceptions.Timeout:
                logger.warning(f"Download attempt {retry+1} timed out")
                if retry == 2:
                    logger.error(f"All download attempts failed for {url} - timeout")
                    return None
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {retry+1} failed: {e}")
                if retry == 2:
                    logger.error(f"All download attempts failed for {url}")
                    return None
                time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error downloading book: {e}")
                return None
    
    def process_book(self, text):
        """Extract words from book text"""
        # Simple word extraction
        words = set()
        for line in text.split('\n'):
            for word in line.split():
                # Clean word
                word = word.strip('.,!?;:""()[]{}').lower()
                if word and len(word) > 1 and word.isalpha():
                    words.add(word)
        return words
    
    def run_once(self):
        """Run one book reading cycle"""
        # Load current vocabulary
        vocab = self.load_vocabulary()
        if not vocab:
            vocab = {}
        
        # Download a book
        book_urls = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
            "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
        ]
        
        for url in book_urls:
            text = self.download_book(url)
            if text:
                new_words = self.process_book(text)
                # Add to vocabulary
                new_count = 0
                for word in new_words:
                    if word not in vocab:
                        vocab[word] = {
                            "first_seen": datetime.now().isoformat(),
                            "source": url,
                            "count": 1
                        }
                        new_count += 1
                logger.info(f"📚 Added {new_count} new words from {url}")
                
                # Save updated vocabulary
                self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.vocab_path, 'w') as f:
                    json.dump(vocab, f, indent=2)
                
                # If we got words, consider this a successful cycle
                if new_count > 0:
                    return True
        
        logger.info("No new words added in this cycle")
        return False
    
    def run_continuous(self):
        """Run continuously"""
        logger.info("📖 Book Reader started in continuous mode")
        while True:
            self.books_read += 1
            logger.info(f"📚 Book reading cycle {self.books_read}")
            self.run_once()
            logger.info("⏰ Next book in 1 hour")
            time.sleep(3600)  # Wait 1 hour

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Book Reader Service")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    reader = BookReader()
    
    if args.test:
        logger.info("🧪 Running in TEST mode - one cycle only")
        reader.run_once()
    elif args.continuous:
        logger.info("🔄 Running in CONTINUOUS mode")
        reader.run_continuous()
    else:
        parser.print_help()

# ============================================================================
# PLUGGABLE INTERFACE LAYER - DO NOT MODIFY BELOW THIS LINE
# ============================================================================
# This section adds API endpoints for external systems to connect
# All original code above remains completely unchanged

import json
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, Any

# Memory optimization
import gc
gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
import resource
try:
    # Set soft memory limit
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
except:
    pass

# Clear cache periodically
import threading
import time
def cache_cleaner():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        if hasattr(__import__('torch'), 'mps'):
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
threading.Thread(target=cache_cleaner, daemon=True).start()


# Global reference to the book reader instance
_book_instance = None
_start_time = datetime.now()

class BookReaderAPIHandler(BaseHTTPRequestHandler):
    """API for external systems to query book reader status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get status
            status = {
                "name": "book_reader",
                "running": True,
                "books_read": 0,
                "vocabulary_size": 0,
                "healthy": True,
                "uptime": str(datetime.now() - _start_time)
            }
            
            # Try to get real data if book instance exists
            if _book_instance:
                try:
                    status["books_read"] = getattr(_book_instance, 'books_read', 0)
                    # Try to load vocabulary to get size
                    vocab = _book_instance.load_vocabulary()
                    if vocab:
                        status["vocabulary_size"] = len(vocab)
                except:
                    pass
            
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/vocabulary':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Return vocabulary statistics
            vocab_stats = {
                "total_words": 0,
                "recent_additions": []
            }
            
            if _book_instance:
                try:
                    vocab = _book_instance.load_vocabulary()
                    if vocab:
                        vocab_stats["total_words"] = len(vocab)
                        # Get 10 most recent words
                        recent = sorted(
                            vocab.items(), 
                            key=lambda x: x[1].get('first_seen', ''), 
                            reverse=True
                        )[:10]
                        vocab_stats["recent_additions"] = [
                            {"word": w, "source": d.get('source', '')} 
                            for w, d in recent
                        ]
                except:
                    pass
            
            self.wfile.write(json.dumps(vocab_stats).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                command = json.loads(post_data)
                cmd = command.get('command', '')
                
                if cmd == 'read_now':
                    # Trigger an immediate book read
                    if _book_instance:
                        result = _book_instance.run_once()
                        self.wfile.write(json.dumps({
                            "status": "read_completed", 
                            "success": result
                        }).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "Book reader not initialized"}).encode())
                elif cmd == 'get_word':
                    word = command.get('word', '')
                    if word and _book_instance:
                        vocab = _book_instance.load_vocabulary()
                        if vocab and word in vocab:
                            self.wfile.write(json.dumps({
                                "found": True,
                                "data": vocab[word]
                            }).encode())
                        else:
                            self.wfile.write(json.dumps({"found": False}).encode())
                    else:
                        self.wfile.write(json.dumps({"error": "No word specified"}).encode())
                else:
                    self.wfile.write(json.dumps({"error": f"Unknown command: {cmd}"}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return  # Suppress HTTP logs

def _start_api_server():
    """Start API server in background thread"""
    port = 9004  # Fixed port for book reader
    
    def run_server():
        server = HTTPServer(('localhost', port), BookReaderAPIHandler)
        print(f"📡 Book Reader API endpoint active at http://localhost:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return port

# Initialize the API server when this module is imported
_api_port = _start_api_server()

# Store reference to book instance when created
_original_init = BookReader.__init__
def _wrapped_init(self, *args, **kwargs):
    global _book_instance
    _original_init(self, *args, **kwargs)
    _book_instance = self

BookReader.__init__ = _wrapped_init
