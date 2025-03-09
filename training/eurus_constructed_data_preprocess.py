'''
Preprocessing on EURUS constructed dataset (dataset that underwent dataseet construction)
After data construction:
1. combine json files into a single json file according to customizable rules
2. convert json file into parquet file using this script
'''

import os
import datasets

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default='dataset/eurus_data_constructed')
    parser.add_argument('--train_data_json', type=str, required=True)
    parser.add_argument('--test_data_json', type=str, required=True)
    args = parser.parse_args()
    
    train_dataset = datasets.load_dataset('json', data_files=args.train_data_json)
    test_dataset = datasets.load_dataset('json', data_files=args.test_data_json)
    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
