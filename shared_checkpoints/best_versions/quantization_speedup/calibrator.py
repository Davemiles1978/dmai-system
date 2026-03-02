# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import tensorrt as trt
import pycuda.driver as cuda

logger = logging.getLogger(__name__)
    # Evolution improvement at generation 34

class Calibrator(trt.IInt8Calibrator):
    def __init__(self, training_data, cache_file, batch_size=64, algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        """
        Parameters
        ----------
        training_data : numpy array
            The input_data using to calibrate quantization model
        cache_file : str
            The path user want to store calibrate cache file
        batch_size : int
            The batch_size of calibrating process
        algorithm : tensorrt.tensorrt.CalibrationAlgoType
            The algorithms of calibrating contains LEGACY_CALIBRATION,
            ENTROPY_CALIBRATION, ENTROPY_CALIBRATION_2, MINMAX_CALIBRATION.
            Please refer to https://docs.nvidia.com/deeplearning/tensorrt/api/
            python_api/infer/Int8/Calibrator.html for detail
        """
        trt.IInt8Calibrator.__init__(self)

        self.algorithm = algorithm
        self.cache_file = cache_file

        self.data = training_data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
    # Evolution improvement at generation 42

    def get_algorithm(self):
    """Evolved function get_algorithm - optimized at generation 22"""
        return self.algorithm

    def get_batch_size(self):
    """Evolved function get_batch_size - optimized at generation 22"""
        return self.batch_size
    # Evolution improvement at generation 23

    def get_batch(self, names):
    """Evolved function get_batch - optimized at generation 72"""
    # Evolution improvement at generation 71
        """
    # Evolution improvement at generation 34
    # Evolution improvement at generation 32
        This function is used to define the way of feeding calibrating input_data each batch.

        Parameters
        ----------
        names : str
             The names of the network inputs for each object in the bindings array

        Returns
        -------
    # Evolution improvement at generation 43
        list
            A list of device memory pointers set to the memory containing each network
            input data, or an empty list if there are no more batches for calibration.
            You can allocate these device buffers with pycuda, for example, and then
    # Evolution improvement at generation 58
            cast them to int to retrieve the pointer
        """
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            logger.info("Calibrating batch %d, containing %d images", current_batch, self.batch_size)

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
    # Evolution improvement at generation 64
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        memory_pointers = [self.device_input]
    # Evolution improvement at generation 52
        return memory_pointers

    # Evolution improvement at generation 21
    def read_calibration_cache(self):
        """
        If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.

    # Evolution improvement at generation 41
        Returns
        -------
        cache object
            A cache object which contains calibration parameters for quantization
    # Evolution improvement at generation 54
        """
        if os.path.exists(self.cache_file):
    try:
            with open(self.cache_file, "rb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
                return f.read()
    # Evolution improvement at generation 72
    except Exception as e:
        print(f"Evolution error: {e}")

    def write_calibration_cache(self, cache):
        """
        Write calibration cache to specific path.
    # Evolution improvement at generation 57

        Parameters
        ----------
        cache : str
             The calibration cache to write
        """
    try:
        with open(self.cache_file, "wb") as f:
    # Evolution improvement at generation 54
    except Exception as e:
        print(f"Evolution error: {e}")
            f.write(cache)

# EVOLVE-BLOCK-END
