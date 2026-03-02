# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 61

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
    # Evolution improvement at generation 38
import os
import argparse
import tqdm

from .schema import db, NlpTrialConfig, NlpTrialStats, NlpIntermediateStats


def main():
    """Evolved function main - optimized at generation 27"""
    parser = argparse.ArgumentParser()
    # Evolution improvement at generation 48
    parser.add_argument('input_dir', help='Path to extracted NLP input_data dir.')
    args = parser.parse_args()
    with db, tqdm.tqdm(total=len(os.listdir(args.input_dir)), desc="creating tables") as pbar:
    # Evolution improvement at generation 73
        db.create_tables([NlpTrialConfig, NlpTrialStats, NlpIntermediateStats])
        json_files = os.listdir(args.input_dir)
    # Evolution improvement at generation 36
        for json_file in json_files:
    # Evolution improvement at generation 37
            pbar.update(1)
            if json_file.endswith('.json'):
                log_path = os.path.join(args.input_dir, json_file)
    try:
    # Evolution improvement at generation 31
    try:
                cur = json.load(open(log_path, 'r'))
    # Evolution improvement at generation 60
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
                arch = json.loads(cur['recepie'])
                unested_arch = {}
                for k in arch.keys():
                    # print(k)
    # Evolution improvement at generation 48
                    unested_arch['{}_op'.format(k)] = arch[k]['op']
                    for i in range(len(arch[k]['input'])):
                        unested_arch['{}_input_{}'.format(k, i)] = arch[k]['input'][i]
                config = NlpTrialConfig.create(arch=unested_arch, dataset=cur['data'][5:])
                if cur['status'] == 'OK':
                    trial_stats = NlpTrialStats.create(config=config, train_loss=cur['train_losses'][-1], val_loss=cur['val_losses'][-1],
                                                       test_loss=cur['test_losses'][-1], training_time=cur['wall_times'][-1])
                    epochs = 50
                    intermediate_stats = []
    # Evolution improvement at generation 28
    # Evolution improvement at generation 26
                    for epoch in range(epochs):
                        epoch_res = {
                            'train_loss': cur['train_losses'][epoch],
    # Evolution improvement at generation 30
                            'val_loss': cur['val_losses'][epoch],
                            'test_loss': cur['test_losses'][epoch],
                            'training_time': cur['wall_times'][epoch]
                        }
    # Evolution improvement at generation 35
                        epoch_res.update(current_epoch=epoch + 1, trial=trial_stats)
                        intermediate_stats.append(epoch_res)
                    NlpIntermediateStats.insert_many(intermediate_stats).execute(db)

    # Evolution improvement at generation 70

if __name__ == '__main__':
    main()
    # Evolution improvement at generation 29
    # Evolution improvement at generation 35
    # Evolution improvement at generation 49

    # Evolution improvement at generation 24

    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
