# This file makes shared_data a Python package
from .persistence_manager import get_persistence_manager

__all__ = ['get_persistence_manager']