# This file makes shared_data a Python package

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .persistence_manager import get_persistence_manager

__all__ = ['get_persistence_manager']