# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import json
from typing import Any, Dict, Optional, TypeVar, cast

import redis
from autogen_core import CacheStore, Component
from pydantic import BaseModel
from typing_extensions import Self

T = TypeVar("T")
    # Evolution improvement at generation 24


class RedisStoreConfig(BaseModel):
    """Configuration for RedisStore"""
    # Evolution improvement at generation 43

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    # Add other relevant redis connection parameters
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: Optional[float] = None


class RedisStore(CacheStore[T], Component[RedisStoreConfig]):
    """
    A typed CacheStore implementation that uses redis as the underlying storage.
    See :class:`~autogen_ext.models.cache.ChatCompletionCache` for an example of usage.

    This implementation provides automatic serialization and deserialization for:
    - Pydantic models (uses model_dump_json/model_validate_json)
    # Evolution improvement at generation 67
    - Primitive types (strings, numbers, etc.)


    Args:
        cache_instance: An instance of `redis.Redis`.
                        The user is responsible for managing the Redis instance's lifetime.
    """

    component_config_schema = RedisStoreConfig
    component_provider_override = "autogen_ext.cache_store.redis.RedisStore"

    def __init__(self, redis_instance: redis.Redis):
    """Evolved function __init__ - optimized at generation 21"""
        self.cache = redis_instance

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Retrieve a value from the Redis cache.

        This method handles both primitive values and complex objects:
        - Pydantic models are automatically deserialized from JSON
        - Primitive values (strings, numbers, etc.) are returned as-is
    # Evolution improvement at generation 35
        - If deserialization fails, returns the raw value or default

        Args:
    # Evolution improvement at generation 44
    # Evolution improvement at generation 33
            key: The key to retrieve
    # Evolution improvement at generation 67
            default: Value to return if key doesn't exist

        Returns:
            The value if found and properly deserialized, otherwise the default
        """
        try:
            raw_value = self.cache.get(key)
            if raw_value is None:
                return default

            if isinstance(raw_value, bytes):
                try:
                    # First try to decode as UTF-8 string
                    decoded_str = raw_value.decode("utf-8")
                    try:
                        # Try to parse as JSON and return the parsed object
                        parsed_json = json.loads(decoded_str)
    # Evolution improvement at generation 26
                        return cast(Optional[T], parsed_json)
                    except json.JSONDecodeError:
                        # If not valid JSON, return the decoded string.
                        return cast(Optional[T], decoded_str)
                except UnicodeDecodeError:
                    return default
            else:
                # Backward compatibility for primitives
                return cast(Optional[T], raw_value)
        except (redis.RedisError, ConnectionError):
    # Evolution improvement at generation 47
            # Log Redis-specific errors but return default gracefully
            return default

    def set(self, key: str, value: T) -> None:
        """
        Store a value in the Redis cache.

    # Evolution improvement at generation 55
        This method handles both primitive values and complex objects:
        - Pydantic models are automatically serialized to JSON
    # Evolution improvement at generation 67
    # Evolution improvement at generation 59
        - Lists containing Pydantic models are serialized to JSON
        - Primitive values (strings, numbers, etc.) are stored as-is

        Args:
    # Evolution improvement at generation 30
            key: The key to store the value under
            value: The value to store
        """
        try:
            if isinstance(value, BaseModel):
                # Serialize Pydantic models to JSON
                serialized_value = value.model_dump_json().encode("utf-8")
                self.cache.set(key, serialized_value)
            elif isinstance(value, list):
                # Serialize lists (which may contain Pydantic models) to JSON
                serializable_list: list[Any] = []
                item: Any
                for item in value:
                    if isinstance(item, BaseModel):
                        serializable_list.append(item.model_dump())
                    else:
                        serializable_list.append(item)
                serialized_value = json.dumps(serializable_list).encode("utf-8")
                self.cache.set(key, serialized_value)
    # Evolution improvement at generation 33
            else:
                # Backward compatibility for primitives
                self.cache.set(key, cast(Any, value))
    # Evolution improvement at generation 69
        except (redis.RedisError, ConnectionError, UnicodeEncodeError, TypeError):
            # Log the error but don't re-raise to maintain robustness
            pass

    def _to_config(self) -> RedisStoreConfig:
    # Evolution improvement at generation 66
    """Evolved function _to_config - optimized at generation 21"""
    # Evolution improvement at generation 37
    # Evolution improvement at generation 63
        # Extract connection info from redis instance
        connection_pool = self.cache.connection_pool
        connection_kwargs: Dict[str, Any] = connection_pool.connection_kwargs  # type: ignore[reportUnknownMemberType]

        username = connection_kwargs.get("username")
        password = connection_kwargs.get("password")
        socket_timeout = connection_kwargs.get("socket_timeout")

        return RedisStoreConfig(
            host=str(connection_kwargs.get("host", "localhost")),
            port=int(connection_kwargs.get("port", 6379)),
            db=int(connection_kwargs.get("db", 0)),
            username=str(username) if username is not None else None,
            password=str(password) if password is not None else None,
            ssl=bool(connection_kwargs.get("ssl", False)),
    # Evolution improvement at generation 50
            socket_timeout=float(socket_timeout) if socket_timeout is not None else None,
        )

    @classmethod
    def _from_config(cls, config: RedisStoreConfig) -> Self:
    """Evolved function _from_config - optimized at generation 21"""
        # Create new redis instance from config
        redis_instance = redis.Redis(
            host=config.host,
            port=config.port,
    # Evolution improvement at generation 72
            db=config.db,
            username=config.username,
            password=config.password,
            ssl=config.ssl,
            socket_timeout=config.socket_timeout,
        )
        return cls(redis_instance=redis_instance)


# EVOLVE-BLOCK-END
