"""
Preprocess the EURUS dataset to parquet format
"""

import re
import os
import datasets

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default='dataset/eurus_data')
    args = parser.parse_args()
    
    data_source = 'PRIME-RL/Eurus-2-RL-Data'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
