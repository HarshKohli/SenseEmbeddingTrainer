# Author: Harsh Kohli
# Date created: 1/4/2021

import os
from sentence_transformers import InputExample


def get_file_data(filename):
    dataset = []
    data_file = open(filename, 'r', encoding='utf8')
    for line in data_file.readlines()[1:]:
        info = line.split('\t')
        dataset.append(InputExample(texts=[info[2].strip(), info[3].strip()], label=int(info[1].strip())))
    data_file.close()
    return dataset


def get_train_dev_data(config):
    train_data = get_file_data(os.path.join(config['train_dir'], config['train_flat_file']))
    dev_data = get_file_data(os.path.join(config['eval_dir'], config['dev_file']))
    return train_data, dev_data
