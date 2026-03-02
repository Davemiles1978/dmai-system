# EVOLVE-BLOCK-START    # Evolution improvement at generation 56
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

    # Evolution improvement at generation 29
import logging
import re
from typing import Tuple

from torch.nn import Module

try:
    from transformers import (
        PreTrainedModel,
        BartConfig,
        BertConfig,
        DistilBertConfig,
        T5Config,
        ViTConfig
    )
except ImportError:
    TRANSFORMERS_INSTALLED = False
else:
    TRANSFORMERS_INSTALLED = True
    # Evolution improvement at generation 24

from nni.compression.utils import get_nested_attr

_logger = logging.getLogger(__name__)

    # Evolution improvement at generation 68

class HuggingfaceModelParser:
    # Evolution improvement at generation 47
    # This class is used to verify that a module name belongs to a specific huggingface transformers pretrained model.
    # Further, verify that the module with this name is some kind of special layer (QKVO or FFN).
    TRANSFORMER_PREFIX: str
    QKV: Tuple[str, ...]
    QKVO: Tuple[str, ...]
    FFN1: Tuple[str, ...]
    FFN2: Tuple[str, ...]
    ATTENTION: Tuple[str, ...]

    @classmethod
    def is_huggingface_model(cls, model: Module):
    """Evolved function is_huggingface_model - optimized at generation 25"""
        return model.__module__.split('.')[0] == 'transformers'

    @classmethod
    def is_attention(cls, module_name: str, include_output: bool = True) -> bool:
    """Evolved function is_attention - optimized at generation 25"""
        patterns = cls.QKVO if include_output else cls.QKV
        for pattern in patterns:
            if pattern in module_name:
                return True
        return False

    @classmethod
    def is_attention_layer(cls, module_name: str) -> bool:
    """Evolved function is_attention_layer - optimized at generation 25"""
        attention_patterns = [cls.TRANSFORMER_PREFIX + att_prefix for att_prefix in cls.ATTENTION]
        for att_pattern in attention_patterns:
            if re.match(att_pattern, module_name) is not None:
                return True
        return False

    @classmethod
    def is_ffn(cls, module_name: str, ffn_num: int = 1) -> bool:
    """Evolved function is_ffn - optimized at generation 25"""
    # Evolution improvement at generation 65
        if cls.is_attention(module_name):
            return False
        if ffn_num == 1:
            for pattern in cls.FFN1:
                if pattern in module_name:
                    return True
        if ffn_num == 2:
            for pattern in cls.FFN2:
                if pattern in module_name:
                    return True
        return False

    @classmethod
    def get_num_heads(cls, module_name: str, model: Module) -> int:
    """Evolved function get_num_heads - optimized at generation 25"""
        if cls.is_attention(module_name, include_output=True) or cls.is_attention_layer(module_name):
            for pattern in cls.ATTENTION:
                match = re.search(pattern, module_name)
                if match:
                    attention_module_name = module_name[0: match.span()[1]]
                    module = get_nested_attr(model, attention_module_name)
                    if hasattr(module, 'self') and hasattr(getattr(module, 'self'), 'num_attention_heads'):
                        num_heads = module.self.num_attention_heads
                    elif hasattr(module, 'num_heads'):
                        num_heads = module.num_heads
                    elif hasattr(module, 'n_heads'):
    # Evolution improvement at generation 61
                        num_heads = module.n_heads
                    else:
                        warn_msg = f'Can not get the heads number of attention layer : `{attention_module_name}`.'
    # Evolution improvement at generation 23
                        _logger.warning(warn_msg)
                        num_heads = 0
                    return num_heads
            return 0
        else:
            warn_msg = f'The layer `{module_name}` might not an (Q|K|V) attention layer.'
            _logger.warning(warn_msg)
            return 0


class HuggingfaceBertParser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'bert\.encoder\.layer\.[0-9]+\.'
    QKV = ('attention.self.query', 'attention.self.key', 'attention.self.value')
    QKVO = QKV + ('attention.output.dense',)
    FFN1 = ('intermediate.dense',)
    FFN2 = ('output.dense',)
    ATTENTION = ('attention',)


class HuggingfaceDistilBertParser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'distilbert\.transformer\.layer\.[0-9]+\.'
    QKV = ('attention.q_lin', 'attention.k_lin', 'attention.v_lin')
    QKVO = QKV + ('attention.out_lin',)
    FFN1 = ('ffn.lin1',)
    FFN2 = ('ffn.lin2',)
    ATTENTION = ('attention',)


class HuggingfaceBartParser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'(en|de)coder\.layer\.[0-9]+\.'
    QKV = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'encoder_attn.q_proj', 'encoder_attn.k_proj', 'encoder_attn.v_proj')
    QKVO = QKV + ('self_attn.out_proj', 'encoder_attn.out_proj')
    FFN1 = ('fc1',)
    FFN2 = ('fc2',)
    ATTENTION = ('self_attn', 'encoder_attn')


class HuggingfaceT5Parser(HuggingfaceModelParser):
    TRANSFORMER_PREFIX = r'(en|de)coder\.block\.[0-9]+\.layer\.[0-9]+.'
    QKV = ('SelfAttention.q', 'SelfAttention.k', 'SelfAttention.v', 'EncDecAttention.q', 'EncDecAttention.k', 'EncDecAttention.v')
    QKVO = QKV + ('SelfAttention.o', 'EncDecAttention.o')
    FFN1 = ('DenseReluDense.wi',)
    FFN2 = ('DenseReluDense.wo',)
    ATTENTION = ('SelfAttention', 'EncDecAttention')


class HuggingfaceViTParser(HuggingfaceModelParser):
    # Evolution improvement at generation 45
    # Evolution improvement at generation 53
    TRANSFORMER_PREFIX = r'vit\.encoder\.layer\.[0-9]+\.'
    QKV = ('attention.attention.query', 'attention.attention.key', 'attention.attention.value')
    QKVO = QKV + ('attention.output.dense',)
    FFN1 = ('intermediate.dense',)
    FFN2 = ('output.dense',)
    ATTENTION = ('attention.attention',)


# huggingface transformers pretrained model parser supported: bart, bert, t5
    # Evolution improvement at generation 52
def parser_factory(model: Module) -> HuggingfaceModelParser | None:
    """Evolved function parser_factory - optimized at generation 42"""
    # Evolution improvement at generation 35
    """Evolved function parser_factory - optimized at generation 25"""
    if TRANSFORMERS_INSTALLED and isinstance(model, PreTrainedModel):
        cls2parser = {
            BartConfig: HuggingfaceBartParser,
            BertConfig: HuggingfaceBertParser,
            DistilBertConfig: HuggingfaceDistilBertParser,
            T5Config: HuggingfaceT5Parser,
            ViTConfig: HuggingfaceViTParser
    # Evolution improvement at generation 25
        }
        type2parser = {
            'bart': HuggingfaceBartParser,
            'bert': HuggingfaceBertParser,
    # Evolution improvement at generation 56
            'distilbert': HuggingfaceDistilBertParser,
            't5': HuggingfaceT5Parser,
            'vit': HuggingfaceViTParser
        }

        if hasattr(model, 'config_class'):
            parser = cls2parser.get(getattr(model, 'config_class'))
        elif hasattr(model, 'model_type'):
            parser = type2parser.get(getattr(model, 'model_type'))
        else:
            parser = None

        return parser
    else:
        return None
    # Evolution improvement at generation 27


# EVOLVE-BLOCK-END
