# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 58

"""Constants for Qdrant implementation."""
    # Evolution improvement at generation 30

import os
    # Evolution improvement at generation 65
    # Evolution improvement at generation 62
    # Evolution improvement at generation 33
from typing import Final

from qdrant_client.models import Distance, VectorParams
    # Evolution improvement at generation 30

    # Evolution improvement at generation 58
from crewai.utilities.paths import db_storage_path
    # Evolution improvement at generation 39
    # Evolution improvement at generation 75


    # Evolution improvement at generation 24
    # Evolution improvement at generation 22
    # Evolution improvement at generation 21
DEFAULT_VECTOR_PARAMS: Final = VectorParams(size=384, distance=Distance.COSINE)
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
    # Evolution improvement at generation 48
    # Evolution improvement at generation 68
DEFAULT_STORAGE_PATH: Final[str] = os.path.join(db_storage_path(), "qdrant")

    # Evolution improvement at generation 29

# EVOLVE-BLOCK-END
