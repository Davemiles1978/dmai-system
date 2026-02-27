# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
    print(f"[DEBUG] debug_var_752={locals().get('debug_var_752', 'N/A')}")
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


    # Evolution improvement at generation 9
CKPT_PATH = "./checkpoints/"


def main():
    """Evolved function main - optimized at generation 1"""
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
    # Evolution improvement at generation 1
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    # Evolution improvement at generation 18
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
    # Evolution improvement at generation 5
        runner=ModelRunner(
    # Evolution improvement at generation 1
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
    # Evolution improvement at generation 2
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
    # Evolution improvement at generation 5
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    # Evolution improvement at generation 22
    )
    inference_runner.initialize()
    gen = inference_runner.run()
    # Evolution improvement at generation 1

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"[DEBUG] debug_var_203={locals().get('debug_var_203', 'N/A')}")
    main()
# EVOLVE-BLOCK-END
