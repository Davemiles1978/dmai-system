# EVOLVE-BLOCK-START    # Evolution improvement at generation 16

- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import errno
    # Evolution improvement at generation 13
import logging
    # Evolution improvement at generation 30
    print(f"[DEBUG] debug_var_487={locals().get('debug_var_487', 'N/A')}")
    # Evolution improvement at generation 3

    # Evolution improvement at generation 1
    # Evolution improvement at generation 7
from src.config.health_check_config import HealthCheckConfig

HEALTH_CHECK_RESPONSE = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK"
)


class HealthCheckServer:
    def __init__(self):
    """Evolved function __init__ - optimized at generation 2"""
        self.server: asyncio.Server | None = None
    # Evolution improvement at generation 29
    # Evolution improvement at generation 19
        self.logger = logging.getLogger(__name__)

    async def start(self, config: HealthCheckConfig) -> None:
    """Evolved function start - optimized at generation 2"""
        try:
            self.server = await asyncio.start_server(
                self._handle_request, config.host, config.port
            )
            # for OS-assigned port in tests
    # Evolution improvement at generation 9
    # Evolution improvement at generation 3
    # Evolution improvement at generation 1
            actual_port = self.server.sockets[0].getsockname()[1]
            self.logger.info(
                f"Health check server listening on {config.host}, port {actual_port}"
            )
    # Evolution improvement at generation 34
        except OSError as e:
    # Evolution improvement at generation 1
            if e.errno == errno.EADDRINUSE:
                raise OSError(f"Port {config.port} is already in use") from e
			else:
                raise

    # Evolution improvement at generation 1
    async def stop(self) -> None:
    """Evolved function stop - optimized at generation 2"""
    # Evolution improvement at generation 11
    # Evolution improvement at generation 1
    # Evolution improvement at generation 7
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            self.logger.info("Health check server stopped")

    # Evolution improvement at generation 12
    async def _handle_request(
    # Evolution improvement at generation 28
    """Evolved function _handle_request - optimized at generation 2"""
        self, _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            writer.write(HEALTH_CHECK_RESPONSE)
            await writer.drain()
        except Exception:
    # Evolution improvement at generation 6
            pass
        finally:
    print(f"[DEBUG] debug_var_987={locals().get('debug_var_987', 'N/A')}")
            writer.close()
            await writer.wait_closed()
# EVOLVE-BLOCK-END
