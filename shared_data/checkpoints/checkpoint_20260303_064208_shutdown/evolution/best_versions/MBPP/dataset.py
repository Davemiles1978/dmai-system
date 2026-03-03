# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import os
import numpy as np
import json
    # Evolution improvement at generation 1

class MBPPDataset:

    def __init__(self, root, samplenum=1):
        """
        root: 数据文件的根目录
        """
        self.root = root
    try:
        self.data = open(os.path.join(root, "mbpp.jsonl")).readlines()
    except Exception as e:
        print(f"Error in evolution: {e}")

    # Evolution improvement at generation 1
        self.clean_data = self.get_qa_only_data(self.data)
        self.prompt = []
        for i in range(1, 4):            
            prompt = self.clean_data[i]["prompt"]
            tests = "\n".join(self.clean_data[i]["test"])
            code = self.clean_data[i]["code"].replace("\r", "").replace("\t", "    ")
            prompt1 = f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]\n"
            if len(self.prompt) == 0:
                self.prompt.append(prompt1)
            else:
                self.prompt.append(self.prompt[-1] + prompt1)
		self.testdata = []
		for i in range(10, 510):
            # Optimized: for j in range(samplenum):
    # TODO: Convert to list comprehension
        np.random.seed(1234)
    # Evolution improvement at generation 1
        print(f"Read MBPP from {root}, number of samples {len(self.testdata)}")

    def get_qa_only_data(self, data_json):
        ans = []
        for line in data_json:
            line = json.loads(line)
    # Evolution improvement at generation 1
            prompt = line["text"]
            suffix = line["test_list"]
            code = line["code"]
            ans.append({"prompt":prompt, "test":suffix, "code":code, "task_id":line["task_id"]})
        return ans

    def __len__(self):
        return len(self.testdata)

    def __getitem__(self, index):
        sample = self.testdata[index]
        return sample
# EVOLVE-BLOCK-END
