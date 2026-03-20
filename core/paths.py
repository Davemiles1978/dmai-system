"""Dynamic path resolution for DMAI system - Works in any environment"""
import os
import sys
from pathlib import Path

class DMAIPaths:
    """Central path management for DMAI"""
    
    @staticmethod
    def get_root():
        """Get the project root directory dynamically"""
        # Start with the directory of this file
        current = Path(__file__).resolve()
        
        # Navigate up to find the project root (where core/ is located)
        for parent in current.parents:
            if (parent / "core").exists() and (parent / "services").exists():
                return parent
        
        # Fallback: go up 1 level from this file
        return current.parent
    
    @staticmethod
    def get_data_path(*subdirs):
        """Get path to data directory"""
        root = DMAIPaths.get_root()
        return root.joinpath("language_learning", "data", *subdirs)
    
    @staticmethod
    def get_secure_path(*subdirs):
        """Get path to secure data directory"""
        root = DMAIPaths.get_root()
        return root.joinpath("language_learning", "data", "secure", *subdirs)
    
    @staticmethod
    def get_vocabulary_path():
        """Get path to vocabulary file"""
        secure_path = DMAIPaths.get_secure_path("vocabulary_master.json")
        if secure_path.exists():
            return secure_path
        
        # Fallback to old location
        old_path = DMAIPaths.get_data_path("vocabulary.json")
        return old_path
    
    @staticmethod
    def get_phrases_path():
        """Get path to phrases file"""
        return DMAIPaths.get_data_path("phrases.json")

# Create convenience instances
paths = DMAIPaths()
ROOT = paths.get_root()
DATA_DIR = paths.get_data_path()
SECURE_DIR = paths.get_secure_path()
VOCAB_PATH = paths.get_vocabulary_path()
PHRASES_PATH = paths.get_phrases_path()

# Add root to Python path if not already there
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
