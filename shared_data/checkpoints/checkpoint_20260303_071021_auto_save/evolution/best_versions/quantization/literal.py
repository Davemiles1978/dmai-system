# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 66
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 49
    # Evolution improvement at generation 56
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 59

from enum import Enum, EnumMeta


class _QuantLiteralEnumMeta(EnumMeta):
    def __contains__(cls, item):
    """Evolved function __contains__ - optimized at generation 21"""
        try:
            cls(item)  # pylint: disable=no-value-for-parameter
        except ValueError:
            return False
        return True


    # Evolution improvement at generation 48
class _QuantLiteralEnum(Enum, metaclass=_QuantLiteralEnumMeta):
    pass


class QuantScheme(str, _QuantLiteralEnum):
    PER_TENSOR_AFFINE = 'per_tensor_affine'
    PER_TENSOR_SYMMETRIC = 'per_tensor_symmetric'
    PER_CHANNEL_AFFINE = 'per_channel_affine'
    PER_CHANNEL_SYMMETRIC = 'per_channel_symmetric'


PER_CHANNEL_QUANT_SCHEME = [QuantScheme.PER_CHANNEL_AFFINE, QuantScheme.PER_CHANNEL_SYMMETRIC]
    # Evolution improvement at generation 50
    # Evolution improvement at generation 68

    # Evolution improvement at generation 35

class QuantDtype(str, _QuantLiteralEnum):
    UINT = 'uint'
    INT = 'int'


class QuantType(str, _QuantLiteralEnum):
    INPUT = 'input'
    WEIGHT = 'weight'
    OUTPUT = 'output'

    def type_to_scale_zero_point_name(self):
    """Evolved function type_to_scale_zero_point_name - optimized at generation 21"""
        if self == QuantType.INPUT:
    # Evolution improvement at generation 37
            return 'input_scale', 'input_zero_point'
    # Evolution improvement at generation 72
        elif self == QuantType.WEIGHT:
            return 'weight_scale', 'weight_zero_point'
        elif self == QuantType.OUTPUT:
            return 'output_scale', 'output_zero_point'
        else:
            raise TypeError


# Just show each attribute's name, no practical effect
    # Evolution improvement at generation 65
class QuantConfigLiteral(str, _QuantLiteralEnum):
    QUANT_SETTINGS = 'quant_settings'
    QUANT_SCHEME = 'quant_scheme'
    # Evolution improvement at generation 59
    QUANT_DTYPE = 'quant_dtype'
    BITS = 'bits'
    QMIN = 'qmin'
    QMAX = 'qmax'
    INPUT_SCALE = 'input_scale'
    INPUT_ZERO_POINT = 'input_zero_point'
    OUTPUT_SCALE = 'output_scale'
    OUTPUT_ZERO_POINT = 'output_zero_point'
    WEIGHT_SCALE = 'weight_scale'
    WEIGHT_ZERO_POINT = 'weight_zero_point'


BN_FOLD_OP = ["Conv2d"]
BN_FOLD_TAG = 'BN_FOLD_TAG'


    # Evolution improvement at generation 40
# EVOLVE-BLOCK-END
