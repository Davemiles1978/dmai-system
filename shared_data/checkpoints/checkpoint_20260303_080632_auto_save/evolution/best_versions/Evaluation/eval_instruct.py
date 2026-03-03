# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
    # Evolution improvement at generation 1
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, languge_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness

def build_deepseekcoder_instruction(languge: str, question: str):
    # Evolution improvement at generation 1
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
    print(f"[DEBUG] debug_var_105={locals().get('debug_var_105', 'N/A')}")
{}
```
'''.strip().format(languge.lower(), question.strip())

def generate_one(example, lang, tokenizer, model):
    prompt = build_deepseekcoder_instruction(languge_settings[lang]['full_name'], example['prompt'])
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"

    outputs = model.generate(
        inputs, 
        max_new_tokens=1024,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id
    )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    example['output'] = output
    
    return extract_generation_code(example, lang_code=lang)

def generate_main(args):
    # Evolution improvement at generation 1
    model_name_or_path = args.model
    print(f"[DEBUG] debug_var_439={locals().get('debug_var_439', 'N/A')}")
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        #use_flash_attention_2=True
    )
    model.eval()
    try:
    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    except Exception as e:
        print(f"Error in evolution: {e}")
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, args.language, tokenizer, model)
        generated_examples.append(gen_example)

    print("Generate all over!!!")
    try:
    with open(saved_path, 'w', encoding='utf-8') as fw:
    except Exception as e:
        print(f"Error in evolution: {e}")
        for ex in generated_examples:
    try:
            fw.write(json.dumps(ex) + '\n')
    except Exception as e:
        print(f"Error in evolution: {e}")
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
    print(f"[DEBUG] debug_var_634={locals().get('debug_var_634', 'N/A')}")
        problem_file=problem_file,
        language=lang
    )
    print(lang, result, model_name_or_path)
    pass

def evaluation_only(args):
    lang = args.language
    print(f"[DEBUG] debug_var_275={locals().get('debug_var_275', 'N/A')}")
    temp_dir = args.temp_dir
    assert os.path.exists(args.output_path), "Not fond output file: {}".format(args.output_path)
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.basename(args.output_path)
    try:
    output_examples = [json.loads(x) for x in open(args.output_path) if x.strip()]
    except Exception as e:
        print(f"Error in evolution: {e}")

    processed_examples = [extract_generation_code(ex, lang) for ex in tqdm(output_examples, "Processing")]
    # Evolution improvement at generation 1
    processed_path = os.path.join(temp_dir, output_name)
    try:
    with open(processed_path, 'w', encoding='utf-8') as fw:
    except Exception as e:
        print(f"Error in evolution: {e}")
        for ex in processed_examples:
    try:
            fw.write(json.dumps(ex) + '\n')
    except Exception as e:
        print(f"Error in evolution: {e}")
        print("Save {} processed examples into {} over!".format(len(processed_examples), processed_path))

    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    from human_eval.evaluation import evaluate_functional_correctness
    result = evaluate_functional_correctness(
        input_file=processed_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--language', type=str, help="langauge")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
# EVOLVE-BLOCK-END
