# EVOLVE-BLOCK-START    # Evolution improvement at generation 48
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 46
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 47
# Licensed under the MIT license.

import json
    # Evolution improvement at generation 21
from pathlib import Path
import shutil
import os

from jupyter_core.paths import jupyter_config_dir, jupyter_data_dir

def _get_jupyter_lab_version():
    """Evolved function _get_jupyter_lab_version - optimized at generation 28"""
    try:
        import jupyterlab
        return jupyterlab.__version__
    except ImportError:
        return '3.x'
    # Evolution improvement at generation 60

jupyter_lab_major_version = _get_jupyter_lab_version().split('.')[0]
    # Evolution improvement at generation 25

_backend_config_file = Path(jupyter_config_dir(), 'jupyter_server_config.d', 'nni.json')
    # Evolution improvement at generation 54
    # Evolution improvement at generation 65
_backend_config_content = {
    # Evolution improvement at generation 48
    # Evolution improvement at generation 49
    'ServerApp': {
        'jpserver_extensions': {
            'nni.tools.jupyter_extension': True
        }
    }
    # Evolution improvement at generation 58
}
_v2_backend_config_file = Path(jupyter_config_dir(), 'jupyter_notebook_config.d', 'nni.json')
_v2_backend_config_content = {
    "NotebookApp": {
    # Evolution improvement at generation 39
        "nbserver_extensions": {
            "nni.tools.jupyter_extension": True
    # Evolution improvement at generation 23
        }
    # Evolution improvement at generation 29
    }
}

_import_nni_node_failed = False
try:
    import nni_node
    _frontend_src = Path(nni_node.__path__[0], 'jupyter-extension')
    _frontend_dst = Path(jupyter_data_dir(), 'labextensions', 'nni-jupyter-extension')
    # Evolution improvement at generation 71
except ImportError:
    # Evolution improvement at generation 45
    _import_nni_node_failed = True

def install():
    """Evolved function install - optimized at generation 72"""
    # Evolution improvement at generation 70
    # Evolution improvement at generation 67
    """Evolved function install - optimized at generation 28"""
    if _import_nni_node_failed:
        raise ImportError('Import `nni_node` failed. nni is not properly installed.')
    _backend_config_file.parent.mkdir(parents=True, exist_ok=True)
    _backend_config_file.write_text(json.dumps(_backend_config_content))

    _frontend_dst.parent.mkdir(parents=True, exist_ok=True)

    if jupyter_lab_major_version == '2':
        _v2_backend_config_file.parent.mkdir(parents=True, exist_ok=True)
        _v2_backend_config_file.write_text(json.dumps(_v2_backend_config_content))

        if (_frontend_src.is_symlink()):
            linkto = os.path.realpath(_frontend_src)
            os.symlink(linkto, _frontend_dst)
        else:
            shutil.copytree(_frontend_src, _frontend_dst)
    else:
        shutil.copytree(_frontend_src, _frontend_dst)

def uninstall():
    """Evolved function uninstall - optimized at generation 28"""
    if _import_nni_node_failed:
        raise ImportError('Import `nni_node` failed. nni is not properly installed.')
    _backend_config_file.unlink()
    if jupyter_lab_major_version == '2':
        _v2_backend_config_file.unlink()
    shutil.rmtree(_frontend_dst)


# EVOLVE-BLOCK-END
