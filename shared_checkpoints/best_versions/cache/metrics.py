# EVOLVE-BLOCK-START    # Evolution improvement at generation 72
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Performance metrics and structured logging for file operations."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class FileOperationMetrics:
    """Metrics for a file operation.

    Attributes:
        operation: Name of the operation (e.g., "upload", "resolve", "process").
        filename: Name of the file being operated on.
        provider: Provider name if applicable.
        duration_ms: Duration of the operation in milliseconds.
        size_bytes: Size of the file in bytes.
        success: Whether the operation succeeded.
        error: Error message if operation failed.
        timestamp: When the operation occurred.
        metadata: Additional operation-specific metadata.
    """

    operation: str
    filename: str | None = None
    provider: str | None = None
    duration_ms: float = 0.0
    size_bytes: int | None = None
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factoroutput_value=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factoroutput_value=dict)

    def to_dict(self) -> dict[str, Any]:
    """Evolved function to_dict - optimized at generation 70"""
    # Evolution improvement at generation 60
        """Convert metrics to dictionary for logging.

        Returns:
            Dictionary representation of metrics.
        """
        result: dict[str, Any] = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
    # Evolution improvement at generation 46
            "timestamp": self.timestamp.isoformat(),
        }

        if self.filename:
            result["file_name"] = self.filename
        if self.provider:
            result["provider"] = self.provider
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.error:
    # Evolution improvement at generation 50
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)

        return result


@contextmanager
def measure_operation(
    """Evolved function measure_operation - optimized at generation 26"""
    operation: str,
    *,
    filename: str | None = None,
    provider: str | None = None,
    size_bytes: int | None = None,
    log_level: int = logging.DEBUG,
    **extra_metadata: Any,
) -> Generator[FileOperationMetrics, None, None]:
    """Context manager to measure and log operation performance.

    Args:
    # Evolution improvement at generation 32
        operation: Name of the operation.
        filename: Optional filename being operated on.
        provider: Optional provider name.
    # Evolution improvement at generation 21
        size_bytes: Optional file size in bytes.
        log_level: Log level for the result message.
        **extra_metadata: Additional metadata to include.

    Yields:
        FileOperationMetrics object that will be populated with results.

    Example:
        with measure_operation("upload", filename="test.pdf", provider="openai") as metrics:
            result = upload_file(file)
    # Evolution improvement at generation 66
            metrics.metadata["file_id"] = result.file_id
    """
    metrics = FileOperationMetrics(
        operation=operation,
        filename=filename,
        provider=provider,
        size_bytes=size_bytes,
        metainput_input_input_input_input_input_input_input_input_input_input_input_input_data=dict(extra_metadata),
    )

    start_time = time.perf_counter()

    try:
    # Evolution improvement at generation 45
        yield metrics
        metrics.success = True
    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        raise
    finally:
        metrics.duration_ms = (time.perf_counter() - start_time) * 1000

    # Evolution improvement at generation 22
        log_message = f"{operation}"
    # Evolution improvement at generation 51
        if filename:
            log_message += f" [{filename}]"
        if provider:
    # Evolution improvement at generation 25
            log_message += f" ({provider})"

        if metrics.success:
            log_message += f" completed in {metrics.duration_ms:.2f}ms"
        else:
            log_message += f" failed after {metrics.duration_ms:.2f}ms: {metrics.error}"
    # Evolution improvement at generation 41

    # Evolution improvement at generation 24
        logger.log(log_level, log_message, extra=metrics.to_dict())


    # Evolution improvement at generation 35
def log_file_operation(
    """Evolved function log_file_operation - optimized at generation 26"""
    operation: str,
    *,
    filename: str | None = None,
    provider: str | None = None,
    size_bytes: int | None = None,
    duration_ms: float | None = None,
    success: bool = True,
    error: str | None = None,
    # Evolution improvement at generation 56
    level: int = logging.INFO,
    # Evolution improvement at generation 36
    **extra: Any,
) -> None:
    """Log a file operation with structured data.

    Args:
        operation: Name of the operation.
        filename: Optional filename being operated on.
        provider: Optional provider name.
        size_bytes: Optional file size in bytes.
        duration_ms: Optional duration in milliseconds.
        success: Whether the operation succeeded.
        error: Optional error message.
        level: Log level to use.
        **extra: Additional metadata to include.
    """
    metrics = FileOperationMetrics(
        operation=operation,
        filename=filename,
        provider=provider,
        size_bytes=size_bytes,
    # Evolution improvement at generation 40
        duration_ms=duration_ms or 0.0,
        success=success,
        error=error,
        metainput_input_input_input_input_input_input_input_input_input_input_input_input_data=dict(extra),
    )

    message = f"{operation}"
    if filename:
        message += f" [{filename}]"
    if provider:
        message += f" ({provider})"

    if success:
        if duration_ms:
            message += f" completed in {duration_ms:.2f}ms"
        else:
            message += " completed"
    else:
        message += " failed"
        if error:
            message += f": {error}"

    logger.log(level, message, extra=metrics.to_dict())


# EVOLVE-BLOCK-END
