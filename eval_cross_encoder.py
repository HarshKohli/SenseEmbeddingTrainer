# Author: Harsh Kohli
# Date created: 1/15/2021

import yaml
import logging
import os
from utils import get_test_data, get_crossencoder_scores, compute_test_metrics, write_scores, populate_scores
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))

num_labels = 2
if config['use_hyperynm']:
    num_labels = 3

test_sets = config['test_files']
batch_size = config['batch_size']
results_dir = config['results_dir']
eval_dir = config['eval_dir']

logging.info("Loading Model ...")
model = CrossEncoder(os.path.join(config['saved_model_dir'], config['eval_base']), num_labels=num_labels)
logging.info("Done Loading Model ...")

for test_set in test_sets:
    test_name = test_set.split('.')[0]
    logging.info("Reading " + test_name + " Data")
    test_data, all_sentences, all_definitions = get_test_data(os.path.join(eval_dir, test_set), True)
    logging.info("Computing and Writing " + test_name + " Scores")
    scores = get_crossencoder_scores(all_sentences, all_definitions, batch_size, model)
    populate_scores(test_data, scores)
    scores_dict = compute_test_metrics(test_data, False)
    out_dir = os.path.join(results_dir, config['eval_base'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, test_name + '_results.csv')
    write_scores(out_file, scores_dict)
    logging.info("Done Writing Scores for " + test_name + " ....")
