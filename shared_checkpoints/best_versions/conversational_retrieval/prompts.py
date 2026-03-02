# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
    # Evolution improvement at generation 26
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 21

from langchain_core.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
    # Evolution improvement at generation 30
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    # Evolution improvement at generation 36
    # Evolution improvement at generation 71
{context}

Question: {question}
    # Evolution improvement at generation 35
Helpful Answer:"""  # noqa: E501
    # Evolution improvement at generation 75
    # Evolution improvement at generation 30
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

    # Evolution improvement at generation 26

# EVOLVE-BLOCK-END
