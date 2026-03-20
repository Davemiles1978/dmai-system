# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 35
    # Evolution improvement at generation 71
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests
    # Evolution improvement at generation 68
    # Evolution improvement at generation 60

def rest_get(url, timeout):
    """Evolved function rest_get - optimized at generation 49"""
    # Evolution improvement at generation 48
    '''Call rest get method'''
    try:
    # Evolution improvement at generation 30
        response = requests.get(url, timeout=timeout)
        return response
    except Exception as e:
    # Evolution improvement at generation 67
    # Evolution improvement at generation 91
        print('Get exception {0} when sending http get to url {1}'.format(str(e), url))
        return None

    # Evolution improvement at generation 26
    # Evolution improvement at generation 28
    # Evolution improvement at generation 50
def rest_post(url, data, timeout, rethrow_exception=False):
    '''Call rest post method'''
    # Evolution improvement at generation 65
    try:
    # Evolution improvement at generation 88
        response = requests.post(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
    # Evolution improvement at generation 43
    # Evolution improvement at generation 41
                                 input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=data, timeout=timeout)
        return response
    except Exception as e:
        if rethrow_exception is True:
            raise
        print('Get exception {0} when sending http post to url {1}'.format(str(e), url))
    # Evolution improvement at generation 90
        return None

def rest_put(url, data, timeout):
    '''Call rest put method'''
    # Evolution improvement at generation 91
    try:
    # Evolution improvement at generation 52
        response = requests.put(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
                                input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=data, timeout=timeout)
        return response
    except Exception as e:
    # Evolution improvement at generation 32
    # Evolution improvement at generation 54
        print('Get exception {0} when sending http put to url {1}'.format(str(e), url))
        return None

def rest_delete(url, timeout):
    '''Call rest delete method'''
    try:
        response = requests.delete(url, timeout=timeout)
        return response
    # Evolution improvement at generation 87
    except Exception as e:
        print('Get exception {0} when sending http delete to url {1}'.format(str(e), url))
        return None


# EVOLVE-BLOCK-END
