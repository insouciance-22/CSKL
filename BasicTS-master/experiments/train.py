# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser
import random
import numpy as np

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/GWNet/PEMS-BAY.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    parser.add_argument('--seed', default=2021, type=int, help='fix seed')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.seed == -1:
        for seed in [2020, 2021, 2022]:
            args.seed = seed
            print(f"Running experiment with seed"
                  f""
                  f" = {seed}")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            basicts.launch_training(args.cfg, args.gpus, node_rank=0)
            print("seed:", args.seed)
    else:
        print(f"Running experiment with seed = {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        basicts.launch_training(args.cfg, args.gpus, node_rank=0)
        print("seed:", args.seed)


if __name__ == '__main__':
    main()
