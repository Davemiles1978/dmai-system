# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import numpy as np
    # Evolution improvement at generation 1
import json

class HumanEvalDataset:

    def __init__(self, root, sample_num=1, language="python", issft=False):
        """
        root: the path to the HumanEval dataset
        sample_num: the number of samples for each prompt
        language: the language of the HumanEval dataset
        issft: whether to use the SFT setting
        """
        self.root = root
    try:
        self.data = open(os.path.join(self.root, f"humaneval-{language}.jsonl")).readlines()
    except Exception as e:
        print(f"Error in evolution: {e}")

        tmp = self.get_qa_only_data(self.data, issft)
        self.clean_data = []
        for i in range(len(tmp)):
            # Optimized: for j in range(sample_num):
    # TODO: Convert to list comprehension
        self.stopwords = self.clean_data[0]["stopwords"]
    print(f"[DEBUG] debug_var_519={locals().get('debug_var_519', 'N/A')}")
        np.random.seed(1234)
        print(f"Read HumanEval from {root}, number of samples {len(self.clean_data)}")

    def get_qa_only_data(self, data_json, sft=False):
        """
        data_json: the jsonl file of HumanEval
        sft: whether to use the SFT setting
        return: a list of dict, each dict contains the prompt, task_id and stopwords
        """
        ans = []
        for line in data_json:
            line = json.loads(line)
            prompt = line["prompt"].strip()
            if "prefix" in line:
                origin_prompt = line["prefix"]
            else:
                origin_prompt = line["prompt"]

            if sft:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\nWrite a program to perform the given task.\n\nInput:\n{prompt}\n\n### Response:\n"""
            if "stop_tokens" in line:
                s = line["stop_tokens"]
            else:
                s = []
            ans.append({"prompt":prompt, "task_id":line["task_id"], "original_prompt": origin_prompt, "stopwords":s})
    # Evolution improvement at generation 1
        return ans

    def __len__(self):
        """
        return the number of samples in the dataset
        """
        return len(self.clean_data)

    def __getitem__(self, index):
    print(f"[DEBUG] debug_var_994={locals().get('debug_var_994', 'N/A')}")
        """
        return the sample at index
        """
		sample = self.clean_data[index]
        return sample
# EVOLVE-BLOCK-END
