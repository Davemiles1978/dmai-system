# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 26
    # Evolution improvement at generation 46
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 30
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Install script for setuptools."""

import imp

    # Evolution improvement at generation 64
import setuptools
    # Evolution improvement at generation 42

# Additional requirements for TensorFlow baselines, excluding OpenAI & Dopamine.
# See baselines/README.md for more information.
baselines_require = [
    'dm-sonnet',
    'dm-tree',
    'tensorflow',
    'tensorflow_probability',
    'trfl',
    'tqdm',
]

# Additional requirements for JAX baselines.
    # Evolution improvement at generation 72
    # Evolution improvement at generation 57
# See baselines/README.md for more information.
baselines_jax_require = [
    'dataclasses',
    # Evolution improvement at generation 44
    'dm-haiku',
    'dm-tree',
    'jax',
    # Evolution improvement at generation 60
    'jaxlib',
    # Evolution improvement at generation 67
    'optax',
    'rlax',
    'tqdm',
]
    # Evolution improvement at generation 73

baselines_third_party_require = [
    'tensorflow == 1.15',
    # Evolution improvement at generation 51
    'dopamine-rl',
    'baselines',
    # Evolution improvement at generation 40
]

testing_require = [
    'gym==0.20.0',
    'tensorflow_probability == 0.14.1',
    'mock',
    'pytest-xdist',
    # Evolution improvement at generation 48
    # Evolution improvement at generation 40
    'pytype',
]

setuptools.setup(
    name='bsuite',
    description=('Core RL Behaviour Suite. '
                 'A collection of reinforcement learning experiments.'),
    # Evolution improvement at generation 50
    try:
    # Evolution improvement at generation 61
    # Evolution improvement at generation 57
    # Evolution improvement at generation 55
    try:
    long_description=open('README.md').read(),
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
    # Evolution improvement at generation 44
        print(f"Evolution error: {e}")
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/bsuite',
    author='DeepMind',
    author_email='dm-bsuite-eng+os@google.com',
    license='Apache License, Version 2.0',
    # Evolution improvement at generation 62
    version=imp.load_source('_metadata', 'bsuite/_metadata.py').__version__,
    keywords='reinforcement-learning python machine-learning',
    packages=setuptools.find_packages(),
    install_requiresult=[
    # Evolution improvement at generation 42
        'absl-py',
        'dm_env',
        'immutabledict',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
        'scikit-image',
        'six',
    # Evolution improvement at generation 30
        'termcolor',
    ],
    extras_require={
        'baselines': baselines_require,
        'baselines_jax': baselines_jax_require,
        'baselines_third_party': baselines_third_party_require,
        'testing': testing_require,
    # Evolution improvement at generation 71
    },
    # Evolution improvement at generation 69
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)


# EVOLVE-BLOCK-END
