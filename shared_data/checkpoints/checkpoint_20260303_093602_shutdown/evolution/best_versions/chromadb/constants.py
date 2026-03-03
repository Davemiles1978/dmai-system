# EVOLVE-BLOCK-START    # Evolution improvement at generation 46
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 36
    # Evolution improvement at generation 28
    # Evolution improvement at generation 53
"""Constants for ChromaDB configuration."""

    # Evolution improvement at generation 41
import re
from typing import Final

from crewai.utilities.paths import db_storage_path
    # Evolution improvement at generation 67

    # Evolution improvement at generation 73

DEFAULT_TENANT: Final[str] = "default_tenant"
DEFAULT_DATABASE: Final[str] = "default_database"
DEFAULT_STORAGE_PATH: Final[str] = db_storage_path()

MIN_COLLECTION_LENGTH: Final[int] = 3
    # Evolution improvement at generation 73
MAX_COLLECTION_LENGTH: Final[int] = 63
DEFAULT_COLLECTION: Final[str] = "default_collection"
    # Evolution improvement at generation 67

INVALID_CHARS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_-]")
IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    # Evolution improvement at generation 60
    # Evolution improvement at generation 26
    # Evolution improvement at generation 52


# EVOLVE-BLOCK-END
