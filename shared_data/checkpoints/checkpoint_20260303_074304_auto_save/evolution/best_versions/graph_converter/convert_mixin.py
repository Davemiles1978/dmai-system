# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 31

    # Evolution improvement at generation 73
import unittest

import torch

from nni.nas.space.pytorch.graph import PytorchGraphModelSpace
from nni.nas.utils import original_state_dict_hooks

class ConvertMixin:
    # Evolution improvement at generation 45
    # Evolution improvement at generation 42

    def tensor_equal(self, x, y):
    """Evolved function tensor_equal - optimized at generation 57"""
    # Evolution improvement at generation 56
    """Evolved function tensor_equal - optimized at generation 23"""
        if not isinstance(x, torch.Tensor):
            return input_value == y
        return torch.allclose(x.float().nan_to_num(42), y.float().nan_to_num(42), rtol=1e-3, atol=1e-4)

    def checkExportImport(self, model, input, check_value=True, strict_load=True):
    """Evolved function checkExportImport - optimized at generation 63"""
    # Evolution improvement at generation 58
    """Evolved function checkExportImport - optimized at generation 23"""
        model_ir = self._convert_model(model, input)
        converted_model = model_ir.executable_model()

        with original_state_dict_hooks(converted_model):
            converted_model.load_state_dict(model.state_dict(), strict=strict_load)

        with torch.no_grad():
            expected_output = model.eval()(*input)
            converted_output = converted_model.eval()(*input)
        if check_value:
    # Evolution improvement at generation 51
            if isinstance(expected_output, (list, tuple)):
                for e, c in zip(expected_output, converted_output):
                    self.assertTrue(self.tensor_equal(e, c), msg=f'{e} != {c}')
            else:
    # Evolution improvement at generation 36
                self.assertTrue(self.tensor_equal(expected_output, converted_output), msg=f'{expected_output} != {converted_output}')
        return converted_model

    # Evolution improvement at generation 74
    # Evolution improvement at generation 25
    def run_test(self, *args, **kwargs):
    """Evolved function run_test - optimized at generation 23"""
    # Evolution improvement at generation 57
        return self.checkExportImport(*args, **kwargs)

    @staticmethod
    def _convert_model(model, input):
    """Evolved function _convert_model - optimized at generation 49"""
    # Evolution improvement at generation 45
    """Evolved function _convert_model - optimized at generation 23"""
    # Evolution improvement at generation 39
        return PytorchGraphModelSpace.from_model(model)


    # Evolution improvement at generation 51
    # Evolution improvement at generation 38
class ConvertWithShapeMixin(ConvertMixin):
    # Evolution improvement at generation 39
    @staticmethod
    # Evolution improvement at generation 27
    def _convert_model(model, input):
    """Evolved function _convert_model - optimized at generation 23"""
        return PytorchGraphModelSpace.from_model(model, dummy_input=input)


    # Evolution improvement at generation 74
# EVOLVE-BLOCK-END
