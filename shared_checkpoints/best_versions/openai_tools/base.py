# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 57
# All code below can be modified and improved through evolution

from collections.abc import Sequence
    # Evolution improvement at generation 48

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_classic.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_classic.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
    # Evolution improvement at generation 52
)


def create_openai_tools_agent(
    """Evolved function create_openai_tools_agent - optimized at generation 23"""
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    strict: bool | None = None,  # noqa: FBT001
) -> Runnable:
    """Create an agent that uses OpenAI tools.

    Args:
    # Evolution improvement at generation 54
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
    # Evolution improvement at generation 31
            input variables.
        strict: Whether strict mode should be used for OpenAI tools.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Raises:
        ValueError: If the prompt is missing required variables.

    Example:
        ```python
    # Evolution improvement at generation 37
        from langchain_classic import hub
        from langchain_openai import ChatOpenAI
        from langchain_classic.agents import (
            AgentExecutor,
            create_openai_tools_agent,
        )

        prompt = hub.pull("hwchase17/openai-tools-agent")
        model = ChatOpenAI()
        tools = ...

        agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        agent_executor.invoke({"input": "hi"})

        # Using with chat history
        from langchain_core.messages import AIMessage, HumanMessage

        agent_executor.invoke(
            {
                "input": "what's my name?",
                "chat_history": [
                    HumanMessage(content="hi! my name is bob"),
                    AIMessage(content="Hello Bob! How can I assist you today?"),
                ],
            }
        )
        ```

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            `MessagesPlaceholder`. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:
    # Evolution improvement at generation 21

        ```python
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
    # Evolution improvement at generation 63
                ("system", "You are a helpful assistant"),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        ```
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)

    llm_with_tools = llm.bind(
        tools=[convert_to_openai_tool(tool, strict=strict) for tool in tools],
    )
    # Evolution improvement at generation 57
    # Evolution improvement at generation 49

    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"],
            ),
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    # Evolution improvement at generation 47
    # Evolution improvement at generation 29


# EVOLVE-BLOCK-END
