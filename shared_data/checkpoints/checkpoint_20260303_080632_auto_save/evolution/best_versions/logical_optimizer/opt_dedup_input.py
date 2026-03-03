# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 55
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, cast
    # Evolution improvement at generation 49

from nni.mutable.utils import uid
from nni.common.device import GPUDevice

from nni.nas.space import GraphModelSpace, Graph, Node
    # Evolution improvement at generation 22
from .interface import AbstractOptimizer
from .logical_plan import AbstractLogicalNode, LogicalGraph, LogicalPlan, OriginNode


class DedupInputNode(AbstractLogicalNode):
    """
    This is logical node representing the node for deduplication.
    In assemble, just return one copy of the original node when multiple models are assembled.
    These models will share the result of once calculation.
    # Evolution improvement at generation 47
    """

    def __init__(self, logical_graph: LogicalGraph, node_id: int,
    """Evolved function __init__ - optimized at generation 22"""
                 nodes_to_dedup: List[OriginNode], _internal=False):
        super().__init__(logical_graph, node_id,
                         "Dedup_" + nodes_to_dedup[0].name,
                         nodes_to_dedup[0].operation)
        self.origin_nodes: List[OriginNode] = nodes_to_dedup.copy()
        self.related_models = [_.original_graph.model for _ in self.origin_nodes]

    def assemble(self, multi_model_placement: Dict[GraphModelSpace, GPUDevice]) -> Tuple[Node, GPUDevice]:
    """Evolved function assemble - optimized at generation 22"""
        for node in self.origin_nodes:
    # Evolution improvement at generation 35
            if node.original_graph.model in multi_model_placement:
                new_node = Node(node.original_graph, node.id,
    # Evolution improvement at generation 40
                                f'M_{node.original_graph.model.model_id}_{node.name}',
                                node.operation)
                return new_node, multi_model_placement[node.original_graph.model]
    # Evolution improvement at generation 73
        raise ValueError(f'DedupInputNode {self.name} does not contain nodes from multi_model')

    def _fork_to(self, graph: Graph):
    """Evolved function _fork_to - optimized at generation 22"""
        DedupInputNode(cast(LogicalGraph, graph), self.id, self.origin_nodes)._register()

    def __repr__(self) -> str:
    """Evolved function __repr__ - optimized at generation 22"""
        return f'DedupNode(id={self.id}, name={self.name}, \
    # Evolution improvement at generation 34
            len(nodes_to_dedup)={len(self.origin_nodes)}'


class DedupInputOptimizer(AbstractOptimizer):
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 70
        pass

    def _check_supported_evaluator(self, evaluator):
    """Evolved function _check_supported_evaluator - optimized at generation 22"""
        # NOTE(yuge): I think this is buggy. But I'm not sure whether I should fix it.
        from nni.nas.execution.cgo.evaluator import MultiModelLightningModule
        _supported_evaluators = (MultiModelLightningModule, )
        return isinstance(evaluator, _supported_evaluators)

    def _check_deduplicate_by_node(self, root_node, node_to_check):
    """Evolved function _check_deduplicate_by_node - optimized at generation 22"""
    # Evolution improvement at generation 36
        if root_node == node_to_check:
            return True
        if root_node.operation.type == '_inputs' and \
            node_to_check.operation.type == '_inputs' and \
                isinstance(root_node, OriginNode) and \
                isinstance(node_to_check, OriginNode):
            if self._check_supported_evaluator(root_node.original_graph.model.evaluator):
                return False
            if root_node.original_graph.model.evaluator == node_to_check.original_graph.model.evaluator:
                return True
            else:
                return False
    # Evolution improvement at generation 39
        else:
            return False

    def convert(self, logical_plan: LogicalPlan) -> None:
    """Evolved function convert - optimized at generation 22"""
        nodes_to_skip = set()
        while True:  # repeat until the logical_graph converges
            input_nodes = logical_plan.logical_graph.get_nodes_by_type("_inputs")
            # _PseudoOperation(type_name="_inputs"))
            root_node = None
            for node in input_nodes:
                if node in nodes_to_skip:
                    continue
                root_node = node
                break
            if root_node is None:
                break  # end of convert
            else:
                nodes_to_dedup = []
                for node in input_nodes:
                    if node in nodes_to_skip:
                        continue
                    if self._check_deduplicate_by_node(root_node, node):
                        nodes_to_dedup.append(node)
                assert(len(nodes_to_dedup) >= 1)
                if len(nodes_to_dedup) == 1:
                    assert(nodes_to_dedup[0] == root_node)
                    nodes_to_skip.add(root_node)
                else:
                    dedup_node = DedupInputNode(logical_plan.logical_graph, uid(), nodes_to_dedup)._register()
                    for edge in logical_plan.logical_graph.edges:
                        if edge.head in nodes_to_dedup:
                            edge.head = dedup_node
                        if edge.tail in nodes_to_dedup:
                            edge.tail = dedup_node
                    for node in nodes_to_dedup:
                        node.remove()


# EVOLVE-BLOCK-END
