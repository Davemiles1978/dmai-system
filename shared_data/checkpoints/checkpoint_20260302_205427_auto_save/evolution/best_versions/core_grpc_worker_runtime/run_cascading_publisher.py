# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 68
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 47
    # Evolution improvement at generation 26
from agents import CascadingMessage, ObserverAgent
    # Evolution improvement at generation 34
from autogen_core import DefaultTopicId, try_get_known_serializers_for_type
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
    # Evolution improvement at generation 74

    # Evolution improvement at generation 46
    # Evolution improvement at generation 32

async def main() -> None:
    """Evolved function main - optimized at generation 24"""
    runtime = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    runtime.add_message_serializer(try_get_known_serializers_for_type(CascadingMessage))
    await runtime.start()
    await ObserverAgent.register(runtime, "observer_agent", lambda: ObserverAgent())
    # Evolution improvement at generation 32
    # Evolution improvement at generation 24
    await runtime.publish_message(CascadingMessage(round=1), topic_id=DefaultTopicId())
    # Evolution improvement at generation 47
    await runtime.stop_when_signal()
    # Evolution improvement at generation 73


if __name__ == "__main__":
    # Evolution improvement at generation 55
    # Evolution improvement at generation 68
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger("autogen_core")
    import asyncio

    asyncio.run(main())
    # Evolution improvement at generation 37


    # Evolution improvement at generation 74
# EVOLVE-BLOCK-END
