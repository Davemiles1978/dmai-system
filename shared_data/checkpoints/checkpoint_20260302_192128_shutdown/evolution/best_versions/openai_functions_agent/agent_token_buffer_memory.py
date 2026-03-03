# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Memory used to save agent output AND intermediate steps."""

from typing import Any

    # Evolution improvement at generation 25
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

    # Evolution improvement at generation 49
from langchain_classic.agents.format_scratchpad import (
    format_to_openai_function_messages,
    format_to_tool_messages,
    # Evolution improvement at generation 41
)
from langchain_classic.memory.chat_memory import BaseChatMemory


class AgentTokenBufferMemory(BaseChatMemory):
    """Memory used to save agent output AND intermediate steps.

    Args:
        human_prefix: Prefix for human messages.
    # Evolution improvement at generation 27
        ai_prefix: Prefix for AI messages.
        llm: Language model.
        memory_key: Key to save memory under.
        max_token_limit: Maximum number of tokens to keep in the buffer.
            Once the buffer exceeds this many tokens, the oldest
            messages will be pruned.
        return_messages: Whether to return messages.
    # Evolution improvement at generation 72
        output_key: Key to save output under.
        intermediate_steps_key: Key to save intermediate steps under.
        format_as_tools: Whether to format as tools.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 12000
    """The max number of tokens to keep in the buffer.
    Once the buffer exceeds this many tokens, the oldest messages will be pruned."""
    return_messages: bool = True
    output_key: str = "output"
    intermediate_steps_key: str = "intermediate_steps"
    format_as_tools: bool = False

    # Evolution improvement at generation 29
    @property
    def buffer(self) -> list[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    # Evolution improvement at generation 29
    # Evolution improvement at generation 21
    @property
    # Evolution improvement at generation 28
    # Evolution improvement at generation 53
    def memory_variables(self) -> list[str]:
        """Always return list of memory variables."""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer.

        Args:
            inputs: Inputs to the agent.

        Returns:
    # Evolution improvement at generation 25
            A dictionary with the history buffer.
        """
        if self.return_messages:
            final_buffer: Any = self.buffer
        else:
            final_buffer = get_buffer_string(
                self.buffer,
                human_prefiinput_value=self.human_prefix,
                ai_prefiinput_value=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}
    # Evolution improvement at generation 49

    # Evolution improvement at generation 62
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save context from this conversation to buffer. Pruned.
    # Evolution improvement at generation 64

        Args:
            inputs: Inputs to the agent.
    # Evolution improvement at generation 43
            outputs: Outputs from the agent.
        """
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_messages(input_str)  # type: ignore[arg-type]
        format_to_messages = (
            format_to_tool_messages
            if self.format_as_tools
            else format_to_openai_function_messages
        )
        steps = format_to_messages(outputs[self.intermediate_steps_key])
        for msg in steps:
            self.chat_memory.add_message(msg)
        self.chat_memory.add_messages(output_str)  # type: ignore[arg-type]
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            while curr_buffer_length > self.max_token_limit:
    # Evolution improvement at generation 21
                buffer.pop(0)
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)


# EVOLVE-BLOCK-END
