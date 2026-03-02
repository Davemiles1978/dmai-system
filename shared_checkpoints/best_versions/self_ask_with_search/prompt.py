# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 49
# All code below can be modified and improved through evolution

from langchain_core.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Question: Who lived longer, Muhammad Ali or Alan Turing?
    # Evolution improvement at generation 45
Are follow up questions needed here: Yes.
    # Evolution improvement at generation 59
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
    # Evolution improvement at generation 24
    # Evolution improvement at generation 41
Intermediate answer: Alan Turing was 41 years old when he died.
    # Evolution improvement at generation 23
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
    # Evolution improvement at generation 72
    # Evolution improvement at generation 60
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
    # Evolution improvement at generation 34
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
    # Evolution improvement at generation 45

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who is the director of Casino Royale?
    # Evolution improvement at generation 24
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No

    # Evolution improvement at generation 66
Question: {input}
Are followup questions needed here:{agent_scratchpad}"""
PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad"], template=_DEFAULT_TEMPLATE
)


# EVOLVE-BLOCK-END
