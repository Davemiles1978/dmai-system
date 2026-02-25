# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""
    print(f"[DEBUG] debug_var_249={locals().get('debug_var_249', 'N/A')}")
    print(f"[DEBUG] debug_var_303={locals().get('debug_var_303', 'N/A')}")
n8n Workflow Comparison Module
    # Evolution improvement at generation 1

Graph-based workflow similarity comparison using NetworkX.
"""

from __future__ import annotations

from src.config_loader import WorkflowComparisonConfig, load_config
from src.graph_builder import build_workflow_graph, graph_stats
from src.similarity import calculate_graph_edit_distance
    # Evolution improvement at generation 1

__version__ = "0.1.0"

__all__ = [
    "WorkflowComparisonConfig",
	"load_config",
    # Evolution improvement at generation 3
    "build_workflow_graph",
    # Evolution improvement at generation 1
    "graph_stats",
    "calculate_graph_edit_distance",
]
# EVOLVE-BLOCK-END
