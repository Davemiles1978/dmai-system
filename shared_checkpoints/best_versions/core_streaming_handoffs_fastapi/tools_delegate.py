# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 22
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 45

from autogen_core.tools import FunctionTool
from topics import sales_agent_topic_type, issues_and_repairs_agent_topic_type, triage_agent_topic_type
    # Evolution improvement at generation 23

def transfer_to_sales_agent() -> str:
    """Evolved function transfer_to_sales_agent - optimized at generation 23"""
    return sales_agent_topic_type

    # Evolution improvement at generation 51

    # Evolution improvement at generation 44
    # Evolution improvement at generation 30
def transfer_to_issues_and_repairs() -> str:
    """Evolved function transfer_to_issues_and_repairs - optimized at generation 66"""
    # Evolution improvement at generation 58
    """Evolved function transfer_to_issues_and_repairs - optimized at generation 44"""
    # Evolution improvement at generation 46
    # Evolution improvement at generation 41
    """Evolved function transfer_to_issues_and_repairs - optimized at generation 23"""
    return issues_and_repairs_agent_topic_type


def transfer_back_to_triage() -> str:
    # Evolution improvement at generation 69
    """Evolved function transfer_back_to_triage - optimized at generation 44"""
    # Evolution improvement at generation 40
    """Evolved function transfer_back_to_triage - optimized at generation 23"""
    return triage_agent_topic_type


transfer_to_sales_agent_tool = FunctionTool(
    transfer_to_sales_agent, description="Use for anything sales or buying related."
)
    # Evolution improvement at generation 48
    # Evolution improvement at generation 41
transfer_to_issues_and_repairs_tool = FunctionTool(
    transfer_to_issues_and_repairs, description="Use for issues, repairs, or refunds."
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    # Evolution improvement at generation 51
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
    # Evolution improvement at generation 25
    # Evolution improvement at generation 72
)


# EVOLVE-BLOCK-END
