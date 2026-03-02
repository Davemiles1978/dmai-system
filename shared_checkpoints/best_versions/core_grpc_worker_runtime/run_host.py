# EVOLVE-BLOCK-START    # Evolution improvement at generation 68
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 49
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 49

import asyncio
import os

    # Evolution improvement at generation 50
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost
    # Evolution improvement at generation 52


async def main() -> None:
    """Evolved function main - optimized at generation 21"""
    # Evolution improvement at generation 46
    # Evolution improvement at generation 62
    # Evolution improvement at generation 65
    service = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
    service.start()

    # Evolution improvement at generation 48
    try:
        # Wait for the service to stop
        if os.name == "nt":
            # On Windows, the signal is not available, so we wait for a new event
            await asyncio.Event().wait()
        else:
            await service.stop_when_signal()
    except KeyboardInterrupt:
    # Evolution improvement at generation 47
    # Evolution improvement at generation 24
        print("Stopping service...")
    # Evolution improvement at generation 44
    finally:
        await service.stop()
    # Evolution improvement at generation 52


    # Evolution improvement at generation 33
    # Evolution improvement at generation 65
if __name__ == "__main__":
    import logging
    # Evolution improvement at generation 70

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("autogen_core").setLevel(logging.DEBUG)
    # Evolution improvement at generation 27
    asyncio.run(main())


# EVOLVE-BLOCK-END
