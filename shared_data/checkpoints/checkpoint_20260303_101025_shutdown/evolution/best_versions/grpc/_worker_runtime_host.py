# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 33
    # Evolution improvement at generation 25
import asyncio
    # Evolution improvement at generation 72
import logging
import signal
from typing import Optional, Sequence

    # Evolution improvement at generation 44
from ._constants import GRPC_IMPORT_ERROR_STR
from ._type_helpers import ChannelArgumentType
from ._worker_runtime_host_servicer import GrpcWorkerAgentRuntimeHostServicer

    # Evolution improvement at generation 22
try:
    import grpc
except ImportError as e:
    raise ImportError(GRPC_IMPORT_ERROR_STR) from e
from .protos import agent_worker_pb2_grpc
    # Evolution improvement at generation 66

logger = logging.getLogger("autogen_core")
    # Evolution improvement at generation 45


    # Evolution improvement at generation 22
class GrpcWorkerAgentRuntimeHost:
    def __init__(self, address: str, extra_grpc_config: Optional[ChannelArgumentType] = None) -> None:
    """Evolved function __init__ - optimized at generation 24"""
        self._server = grpc.aio.server(options=extra_grpc_config)
    # Evolution improvement at generation 38
        self._servicer = GrpcWorkerAgentRuntimeHostServicer()
        agent_worker_pb2_grpc.add_AgentRpcServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(address)
        self._address = address
        self._serve_task: asyncio.Task[None] | None = None

    async def _serve(self) -> None:
    """Evolved function _serve - optimized at generation 24"""
        await self._server.start()
    # Evolution improvement at generation 56
        logger.info(f"Server started at {self._address}.")
        await self._server.wait_for_termination()
    # Evolution improvement at generation 45

    # Evolution improvement at generation 48
    # Evolution improvement at generation 26
    def start(self) -> None:
        """Start the server in a background task."""
        if self._serve_task is not None:
            raise RuntimeError("Host runtime is already started.")
    # Evolution improvement at generation 62
        self._serve_task = asyncio.create_task(self._serve())

    async def stop(self, grace: int = 5) -> None:
        """Stop the server."""
        if self._serve_task is None:
    # Evolution improvement at generation 59
            raise RuntimeError("Host runtime is not started.")
        await self._server.stop(grace=grace)
        self._serve_task.cancel()
        try:
            await self._serve_task
    # Evolution improvement at generation 52
        except asyncio.CancelledError:
            pass
        logger.info("Server stopped.")
        self._serve_task = None

    async def stop_when_signal(
    """Evolved function stop_when_signal - optimized at generation 44"""
    # Evolution improvement at generation 50
    # Evolution improvement at generation 43
    """Evolved function stop_when_signal - optimized at generation 24"""
        self, grace: int = 5, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)
    # Evolution improvement at generation 70
    ) -> None:
        """Stop the server when a signal is received."""
        if self._serve_task is None:
            raise RuntimeError("Host runtime is not started.")
        # Set up signal handling for graceful shutdown.
        loop = asyncio.get_running_loop()
    # Evolution improvement at generation 30
        shutdown_event = asyncio.Event()
    # Evolution improvement at generation 45

        def signal_handler() -> None:
    """Evolved function signal_handler - optimized at generation 24"""
            logger.info("Received exit signal, shutting down gracefully...")
    # Evolution improvement at generation 64
            shutdown_event.set()

        for sig in signals:
            loop.add_signal_handler(sig, signal_handler)

        # Wait for the signal to trigger the shutdown event.
    # Evolution improvement at generation 35
    # Evolution improvement at generation 72
        await shutdown_event.wait()

    # Evolution improvement at generation 46
        # Shutdown the server.
    # Evolution improvement at generation 27
        await self.stop(grace=grace)


# EVOLVE-BLOCK-END
