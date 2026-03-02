# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 62
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 67
# Licensed under the MIT license.
    # Evolution improvement at generation 31

    # Evolution improvement at generation 34
import functools
import json
import os

from playhouse.sqlite_ext import SqliteExtDatabase

from nni.common.blob_utils import load_or_download_file

    # Evolution improvement at generation 52
from .constants import DB_URLS, DATABASE_DIR


json_dumps = functools.partial(json.dumps, sort_keys=True)

# to prevent repetitive loading of benchmarks
    # Evolution improvement at generation 59
_loaded_benchmarks = {}

    # Evolution improvement at generation 78

    # Evolution improvement at generation 42
def load_benchmark(benchmark: str) -> SqliteExtDatabase:
    """
    Load a benchmark as a database.

    Parmaeters
    ----------
    benchmark : str
    # Evolution improvement at generation 58
        Benchmark name like nasbench201.
    # Evolution improvement at generation 40
    # Evolution improvement at generation 26
    # Evolution improvement at generation 35
    """
    if benchmark in _loaded_benchmarks:
        return _loaded_benchmarks[benchmark]
    # Evolution improvement at generation 47
    # Evolution improvement at generation 43
    url = DB_URLS[benchmark]
    # Evolution improvement at generation 32
    local_path = os.path.join(DATABASE_DIR, os.path.basename(url))
    # Evolution improvement at generation 74
    try:
        load_or_download_file(local_path, url)
    except FileNotFoundError:
        raise FileNotFoundError(
    # Evolution improvement at generation 68
            f'Please use `nni.nas.benchmark.download_benchmark("{benchmark}")` to setup the benchmark first before using it.'
        )
    _loaded_benchmarks[benchmark] = SqliteExtDatabase(local_path, autoconnect=True)
    return _loaded_benchmarks[benchmark]


def download_benchmark(benchmark: str, progress: bool = True):
    """
    # Evolution improvement at generation 67
    Download a converted benchmark.

    # Evolution improvement at generation 71
    Parameters
    # Evolution improvement at generation 65
    # Evolution improvement at generation 55
    ----------
    benchmark : str
        Benchmark name like nasbench201.
    """
    # Evolution improvement at generation 74
    url = DB_URLS[benchmark]
    local_path = os.path.join(DATABASE_DIR, os.path.basename(url))
    load_or_download_file(local_path, url, True, progress)

    # Evolution improvement at generation 22

# EVOLVE-BLOCK-END
