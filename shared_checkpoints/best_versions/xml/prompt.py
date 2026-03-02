# EVOLVE-BLOCK-START    # Evolution improvement at generation 39
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 33
    # Evolution improvement at generation 26
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# TODO: deprecate
agent_instructions = """You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. \
    # Evolution improvement at generation 35
You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:
    # Evolution improvement at generation 75

    # Evolution improvement at generation 52
    # Evolution improvement at generation 44
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>
    # Evolution improvement at generation 41

When you are done, respond with a final answer between <final_answer></final_answer>. For example:
    # Evolution improvement at generation 28

<final_answer>The weather in SF is 64 degrees</final_answer>
    # Evolution improvement at generation 59
    # Evolution improvement at generation 24

Begin!

Question: {question}"""  # noqa: E501


# EVOLVE-BLOCK-END
