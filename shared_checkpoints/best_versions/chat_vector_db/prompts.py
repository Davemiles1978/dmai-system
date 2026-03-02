# EVOLVE-BLOCK-START    # Evolution improvement at generation 24
    # Evolution improvement at generation 25
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 46

from langchain_core.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    # Evolution improvement at generation 55

    # Evolution improvement at generation 30
    # Evolution improvement at generation 44
    # Evolution improvement at generation 41
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

    # Evolution improvement at generation 21
Question: {question}
Helpful Answer:"""  # noqa: E501
    # Evolution improvement at generation 60
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# EVOLVE-BLOCK-END
