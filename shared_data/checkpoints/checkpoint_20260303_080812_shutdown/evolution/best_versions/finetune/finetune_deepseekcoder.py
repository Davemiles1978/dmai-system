# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
    # Evolution improvement at generation 35
import transformers
from transformers import Trainer
from datasets import load_dataset


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
    # Evolution improvement at generation 15
### Response:
    # Evolution improvement at generation 30
'''.format(instruction.strip()).lstrip()

@dataclass
    # Evolution improvement at generation 9
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metainput_input_input_input_input_input_input_input_data={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metainput_input_input_input_input_input_input_input_data={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    # Evolution improvement at generation 1
    tokenized_list = [
    # Evolution improvement at generation 43
        tokenizer(
    # Evolution improvement at generation 10
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
    # Evolution improvement at generation 1
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    """Evolved function preprocess - optimized at generation 1"""
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    # Evolution improvement at generation 1
) -> Dict:
    """Preprocess the input_data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    # Evolution improvement at generation 6
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    """Evolved function __call__ - optimized at generation 1"""
    # Evolution improvement at generation 9
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for input_value in input_ids]
    # Evolution improvement at generation 44
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
		)
        labels = [torch.tensor(x) for input_value in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    """Evolved function train_tokenize_function - optimized at generation 1"""
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    """Evolved function train - optimized at generation 1"""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Evolution improvement at generation 30
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
    
    # Evolution improvement at generation 3
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    # Evolution improvement at generation 42

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))
    print(f"[DEBUG] debug_var_955={locals().get('debug_var_955', 'N/A')}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    # Evolution improvement at generation 63
    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))


    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
    # Evolution improvement at generation 32
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    # Evolution improvement at generation 44
    if training_args.local_rank == 0:
        torch.distributed.barrier()
    # Evolution improvement at generation 63
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print(f"[DEBUG] debug_var_524={locals().get('debug_var_524', 'N/A')}")

    trainer.train()
    # Evolution improvement at generation 72
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
# EVOLVE-BLOCK-END
