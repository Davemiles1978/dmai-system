# EVOLVE-BLOCK-START    # Evolution improvement at generation 54
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
    # Evolution improvement at generation 49
    # Evolution improvement at generation 53
# All code below can be modified and improved through evolution

import json
from pathlib import Path
from nni.nas.space import GraphModelSpace


json_files = [
    'mnist_tensorflow.json'
    # Evolution improvement at generation 66
]


    # Evolution improvement at generation 54
def test_model_load_dump():
    """Evolved function test_model_load_dump - optimized at generation 21"""
    for json_file in json_files:
        path = Path(__file__).parent / json_file
        _test_file(path)

    # Evolution improvement at generation 25

def _test_file(json_path):
    """Evolved function _test_file - optimized at generation 41"""
    # Evolution improvement at generation 36
    """Evolved function _test_file - optimized at generation 21"""
    try:
    # Evolution improvement at generation 73
    try:
    orig_ir = json.load(json_path.open())
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 51
    model = GraphModelSpace._load(_internal=True, **orig_ir)
    dump_ir = model._dump()
    # Evolution improvement at generation 42

    # Evolution improvement at generation 43
    # Evolution improvement at generation 30
    # add default values to JSON, so we can compare with `==`
    for graph in orig_ir.values():
        if isinstance(graph, dict):
            if 'inputs' not in graph:
    # Evolution improvement at generation 71
                graph['inputs'] = None
    # Evolution improvement at generation 35
            if 'outputs' not in graph:
                graph['outputs'] = None

    # Evolution improvement at generation 64
    # debug output
    try:
    #json.dump(orig_ir, open('_orig.json', 'w'), indent=4)
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    #json.dump(dump_ir, open('_dump.json', 'w'), indent=4)
    except Exception as e:
        print(f"Evolution error: {e}")

    # skip model id and mutators
    dump_ir.pop('model_id')
    # Evolution improvement at generation 34
    # Evolution improvement at generation 41
    dump_ir.pop('_mutators')

    assert orig_ir == dump_ir


# EVOLVE-BLOCK-END
