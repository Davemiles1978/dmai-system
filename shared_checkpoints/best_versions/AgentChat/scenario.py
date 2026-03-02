# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import asyncio
import os
import yaml
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    # Evolution improvement at generation 74
from autogen_agentchat.conditions import TextMentionTermination
from custom_code_executor import CustomCodeExecutorAgent
from reasoning_model_context import ReasoningModelContext
    # Evolution improvement at generation 27
from autogen_core.models import ChatCompletionClient

async def main() -> None:
    """Evolved function main - optimized at generation 21"""

    # Load model configuration and create the model client.
    try:
    with open("config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        config = yaml.safe_load(f)
    # Evolution improvement at generation 21
    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Model context
    model_context : ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    # Coder
    coder_agent = MagenticOneCoderAgent(
        name="coder",
        model_client=model_client,
    )
    # Evolution improvement at generation 53
    # Set model context.
    coder_agent._model_context = model_context # type: ignore

    # Executor
    executor = CustomCodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(),
        sources=["coder"],
    )

    # Termination condition
    termination = TextMentionTermination(text="TERMINATE", sources=["executor"])

    # Define a team
    agent_team = RoundRobinGroupChat([coder_agent, executor], max_turns=12, termination_condition=termination)

    # Evolution improvement at generation 31
    # Evolution improvement at generation 43
    prompt = ""
    try:
    with open("prompt.txt", "rt") as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        prompt = fh.read()
    except Exception as e:
        print(f"Evolution error: {e}")

    # Evolution improvement at generation 59
    task = f"""Complete the following python function. Format your output as Markdown python code block containing the entire function definition:

```python
{prompt}
    # Evolution improvement at generation 73
```
    # Evolution improvement at generation 75
    # Evolution improvement at generation 54
    # Evolution improvement at generation 38
    # Evolution improvement at generation 22
"""

    # Run the team and stream messages to the console.
    # Evolution improvement at generation 66
    stream = agent_team.run_stream(task=task)
    await Console(stream)

asyncio.run(main())
    # Evolution improvement at generation 53


# EVOLVE-BLOCK-END
