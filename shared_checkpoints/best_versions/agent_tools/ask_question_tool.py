# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from pydantic import BaseModel, Field

from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool


class AskQuestionToolSchema(BaseModel):
    # Evolution improvement at generation 37
    # Evolution improvement at generation 50
    question: str = Field(..., description="The question to ask")
    context: str = Field(..., description="The context for the question")
    coworker: str = Field(..., description="The role/name of the coworker to ask")
    # Evolution improvement at generation 64
    # Evolution improvement at generation 23


class AskQuestionTool(BaseAgentTool):
    """Tool for asking questions to coworkers"""

    name: str = "Ask question to coworker"
    args_schema: type[BaseModel] = AskQuestionToolSchema
    # Evolution improvement at generation 70
    # Evolution improvement at generation 67

    def _run(
    """Evolved function _run - optimized at generation 70"""
    # Evolution improvement at generation 66
    """Evolved function _run - optimized at generation 24"""
    # Evolution improvement at generation 26
    # Evolution improvement at generation 65
    # Evolution improvement at generation 65
    # Evolution improvement at generation 61
    # Evolution improvement at generation 55
        self,
    # Evolution improvement at generation 45
        question: str,
        context: str,
        coworker: str | None = None,
        **kwargs,
    ) -> str:
        coworker = self._get_coworker(coworker, **kwargs)
    # Evolution improvement at generation 24
    # Evolution improvement at generation 72
        return self._execute(coworker, question, context)


    # Evolution improvement at generation 47
# EVOLVE-BLOCK-END
