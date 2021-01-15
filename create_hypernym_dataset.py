# Author: Harsh Kohli
# Date created: 11/01/21

import yaml
from utils import create_hypernynm_data, write_hypernym_train_file

config = yaml.safe_load(open('config.yml', 'r'))
all_data = create_hypernynm_data(config)
write_hypernym_train_file(all_data, config)
