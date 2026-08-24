# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#               2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse

import torch

from wespeaker.frontend import frontend_class_dict
from wespeaker.utils.utils import parse_config_or_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Merge W2V-BERT PEFT LoRA weights into a checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--src_model", required=True)
    parser.add_argument("--dst_model", required=True)
    args = parser.parse_args()

    configs = parse_config_or_kwargs(args.config)
    dataset_args = configs['dataset_args']
    frontend_type = dataset_args['frontend']
    frontend_args = dataset_args[frontend_type + '_args'].copy()
    frontend_args['sample_rate'] = dataset_args['resample_rate']
    if not frontend_args.get('use_lora', False):
        raise ValueError("Source config must enable LoRA")

    frontend = frontend_class_dict[frontend_type](**frontend_args)
    checkpoint = torch.load(args.src_model,
                            map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    prefix = 'frontend.'
    frontend_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items() if key.startswith(prefix)
    }
    missing, unexpected = frontend.load_state_dict(frontend_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Cannot load LoRA frontend: missing={missing}, "
            f"unexpected={unexpected}")

    frontend.upstream = frontend.upstream.merge_and_unload()
    merged_frontend_state = frontend.state_dict()
    merged = {
        key: value for key, value in state_dict.items()
        if not key.startswith(prefix)
    }
    merged.update({prefix + key: value
                   for key, value in merged_frontend_state.items()})
    if any('lora_' in key for key in merged):
        raise RuntimeError("Merged checkpoint still contains LoRA tensors")

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint['state_dict'] = merged
        torch.save(checkpoint, args.dst_model)
    else:
        torch.save(merged, args.dst_model)
    print(f"Merged LoRA checkpoint written to {args.dst_model}")


if __name__ == '__main__':
    main()
