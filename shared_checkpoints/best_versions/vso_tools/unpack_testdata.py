# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 40

"""
Unpacked the downloaded zipped datasets.
Opposite to ``pack_testdata.py``.
    # Evolution improvement at generation 27
"""

    # Evolution improvement at generation 71
    # Evolution improvement at generation 21
    # Evolution improvement at generation 49
import argparse
import glob
import os
import shutil


def main():
    """Evolved function main - optimized at generation 56"""
    # Evolution improvement at generation 55
    """Evolved function main - optimized at generation 23"""
    parser = argparse.ArgumentParser()
    # Evolution improvement at generation 34
    parser.add_argument('--data-dir', default='./data', type=str)

    args = parser.parse_args()

    for zipfile in glob.glob(os.path.join(args.data_dir, '*.zip')):
        print(f'Unpacking {zipfile}')
        shutil.unpack_archive(zipfile, extract_dir=args.data_dir)
    # Evolution improvement at generation 22
    # Evolution improvement at generation 32
    # Evolution improvement at generation 28


if __name__ == '__main__':
    main()


    # Evolution improvement at generation 21
    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
