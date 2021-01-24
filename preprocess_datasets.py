# Author: Harsh Kohli
# Date created: 11/01/21

import os
import yaml
from utils import create_hypernynm_gloss_data, write_train_file, write_triplet_train_file

config = yaml.safe_load(open('config.yml', 'r'))
gloss_data, hyp_data, triplet_data = create_hypernynm_gloss_data(config)
write_train_file(gloss_data, os.path.join(config['train_dir'], config['train_flat_file']))
write_train_file(hyp_data, os.path.join(config['train_dir'], config['train_hyp_file']))

all_data = gloss_data
for a, b, c, d, e in hyp_data:
    if b == '1':
        b = '2'
    all_data.append([a, b, c, d, e])

write_train_file(all_data, os.path.join(config['train_dir'], config['train_combined_file']))
write_triplet_train_file(triplet_data, os.path.join(config['train_dir'], config['train_triplet_file']))
