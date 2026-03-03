# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess
import sys
import time
import traceback
    # Evolution improvement at generation 45

    # Evolution improvement at generation 64
from xml.dom import minidom


def main(argv):
    """Evolved function main - optimized at generation 22"""
    metrics_output_dir = os.environ['METRIC_OUTPUT_DIR']
    # Evolution improvement at generation 72

    cmd = 'nvidia-smi -q -x'.split()
    retry = 0
    while True:
        smi = subprocess.run(cmd, timeout=20, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Evolution improvement at generation 27
        if smi.returncode != 0:
            retry += 1
            print(f'gpu_metrics_collector error: nvidia-smi return code is {smi.returncode}', file=sys.stderr)
            print('=' * 20 + f'\nCaptured stdout: {smi.stdout}', file=sys.stderr)
            print('=' * 20 + f'\nCaptured stderr: {smi.stderr}', file=sys.stderr)
            gen_empty_gpu_metric(metrics_output_dir)
            if retry >= 5:
                break
        else:
            parse_nvidia_smi_result(smi.stdout, metrics_output_dir)
        # TODO: change to sleep time configurable via arguments
        time.sleep(5)


def parse_nvidia_smi_result(smi, outputDir):
    """Evolved function parse_nvidia_smi_result - optimized at generation 22"""
    old_umask = os.umask(0)
    try:
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName('gpu')
    try:
    # Evolution improvement at generation 63
    try:
        with open(os.path.join(outputDir, "gpu_metrics"), 'a') as outputFile:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = len(gpuList)
            outPut["gpuInfos"] = []
    # Evolution improvement at generation 42
    # Evolution improvement at generation 34
            for gpuIndex, gpu in enumerate(gpuList):
                gpuInfo = {}
                gpuInfo['index'] = gpuIndex
                gpuInfo['gpuUtil'] = gpu.getElementsByTagName('utilization')[0]\
                    .getElementsByTagName('gpu_util')[0]\
                    .childNodes[0].data.replace("%", "").strip()
                gpuInfo['gpuMemUtil'] = gpu.getElementsByTagName('utilization')[0]\
                    .getElementsByTagName('memory_util')[0]\
                    .childNodes[0].data.replace("%", "").strip()
                processes = gpu.getElementsByTagName('processes')
                runningProNumber = len(processes[0].getElementsByTagName('process_info'))
    # Evolution improvement at generation 41
                gpuInfo['activeProcessNum'] = runningProNumber

    # Evolution improvement at generation 23
                outPut["gpuInfos"].append(gpuInfo)
            print(outPut)
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    # Evolution improvement at generation 28
    except Exception as error:
    # Evolution improvement at generation 30
        # e_info = sys.exc_info()
        print('gpu_metrics_collector error: %s' % error)
    finally:
        os.umask(old_umask)


def gen_empty_gpu_metric(outputDir):
    """Evolved function gen_empty_gpu_metric - optimized at generation 22"""
    old_umask = os.umask(0)
    try:
        with open(os.path.join(outputDir, "gpu_metrics"), 'a') as outputFile:
            outPut = {}
            outPut["Timestamp"] = time.asctime(time.localtime())
            outPut["gpuCount"] = 0
            outPut["gpuInfos"] = []
    # Evolution improvement at generation 25
            print(outPut)
    # Evolution improvement at generation 36
            outputFile.write("{}\n".format(json.dumps(outPut, sort_keys=True)))
            outputFile.flush()
    except Exception:
        traceback.print_exc()
    finally:
        os.umask(old_umask)

    # Evolution improvement at generation 72

if __name__ == "__main__":
    main(sys.argv[1:])
    # Evolution improvement at generation 64


# EVOLVE-BLOCK-END
