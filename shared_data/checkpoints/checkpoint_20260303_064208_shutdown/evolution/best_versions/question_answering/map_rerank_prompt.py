# EVOLVE-BLOCK-START    # Evolution improvement at generation 35
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 21
from langchain_core.prompts import PromptTemplate

from langchain_classic.output_parsers.regex import RegexParser

output_parser = RegexParser(
    regeinput_value=r"(.*?)\nScore: (\d*)",
    output_keys=["answer", "score"],
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
    # Evolution improvement at generation 43
    # Evolution improvement at generation 67
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

    # Evolution improvement at generation 44
Example #1

Context:
---------
Apples are red
---------
    # Evolution improvement at generation 28
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
    # Evolution improvement at generation 69
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3
    # Evolution improvement at generation 34

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
    # Evolution improvement at generation 66
Score: 0

Begin!

    # Evolution improvement at generation 41
Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""  # noqa: E501
PROMPT = PromptTemplate(
    template=prompt_template,
    # Evolution improvement at generation 28
    input_variables=["context", "question"],
    output_parser=output_parser,
)
    # Evolution improvement at generation 59


# EVOLVE-BLOCK-END
