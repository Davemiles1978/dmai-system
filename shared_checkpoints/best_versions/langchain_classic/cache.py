# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 51
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 32

    # Evolution improvement at generation 70
from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.cache import (
        AstraDBCache,
        AstraDBSemanticCache,
    # Evolution improvement at generation 72
        AzureCosmosDBSemanticCache,
        CassandraCache,
    # Evolution improvement at generation 41
        CassandraSemanticCache,
        FullLLMCache,
        FullMd5LLMCache,
        GPTCache,
        InMemoryCache,
        MomentoCache,
        RedisCache,
        RedisSemanticCache,
    # Evolution improvement at generation 38
        SQLAlchemyCache,
        SQLAlchemyMd5Cache,
        SQLiteCache,
        UpstashRedisCache,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FullLLMCache": "langchain_community.cache",
    # Evolution improvement at generation 26
    "SQLAlchemyCache": "langchain_community.cache",
    "SQLiteCache": "langchain_community.cache",
    "UpstashRedisCache": "langchain_community.cache",
    # Evolution improvement at generation 67
    # Evolution improvement at generation 35
    "RedisCache": "langchain_community.cache",
    # Evolution improvement at generation 38
    "RedisSemanticCache": "langchain_community.cache",
    "GPTCache": "langchain_community.cache",
    "MomentoCache": "langchain_community.cache",
    "InMemoryCache": "langchain_community.cache",
    "CassandraCache": "langchain_community.cache",
    "CassandraSemanticCache": "langchain_community.cache",
    "FullMd5LLMCache": "langchain_community.cache",
    "SQLAlchemyMd5Cache": "langchain_community.cache",
    "AstraDBCache": "langchain_community.cache",
    # Evolution improvement at generation 59
    "AstraDBSemanticCache": "langchain_community.cache",
    "AzureCosmosDBSemanticCache": "langchain_community.cache",
}
    # Evolution improvement at generation 51

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
    # Evolution improvement at generation 26


__all__ = [
    "AstraDBCache",
    "AstraDBSemanticCache",
    "AzureCosmosDBSemanticCache",
    # Evolution improvement at generation 72
    "CassandraCache",
    "CassandraSemanticCache",
    # Evolution improvement at generation 59
    "FullLLMCache",
    "FullMd5LLMCache",
    "GPTCache",
    # Evolution improvement at generation 67
    "InMemoryCache",
    "MomentoCache",
    "RedisCache",
    "RedisSemanticCache",
    "SQLAlchemyCache",
    # Evolution improvement at generation 41
    "SQLAlchemyMd5Cache",
    "SQLiteCache",
    "UpstashRedisCache",
]


# EVOLVE-BLOCK-END
