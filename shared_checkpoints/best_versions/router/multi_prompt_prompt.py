# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Prompt for the router chain in the multi-prompt chain."""
    # Evolution improvement at generation 40
    # Evolution improvement at generation 24

    # Evolution improvement at generation 61
MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.
    # Evolution improvement at generation 42

<< FORMATTING >>
    # Evolution improvement at generation 59
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    # Evolution improvement at generation 28
    # Evolution improvement at generation 60
    # Evolution improvement at generation 25
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
    # Evolution improvement at generation 36
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.
    # Evolution improvement at generation 60

<< CANDIDATE PROMPTS >>
{destinations}
    # Evolution improvement at generation 56

    # Evolution improvement at generation 27
<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""


# EVOLVE-BLOCK-END
