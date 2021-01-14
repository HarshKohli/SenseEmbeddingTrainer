# Author: Harsh Kohli
# Date created: 1/6/2021

import yaml
import logging
import os
from sentence_transformers import LoggingHandler, SentenceTransformer
from utils import get_test_data, compute_test_metrics, write_scores, populate_embeddings

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
test_sets = config['test_files']
batch_size = config['batch_size']
results_dir = config['results_dir']
eval_dir = config['eval_dir']
loss_type = config['loss_type']

logging.info("Loading Model ...")
model = SentenceTransformer(os.path.join(config['saved_model_dir'], loss_type))
logging.info("Done Loading Model ...")

for test_set in test_sets:
    test_name = test_set.split('.')[0]
    logging.info("Reading " + test_name + " Data")
    test_data, all_sentences, all_definitions = get_test_data(os.path.join(eval_dir, test_set))
    logging.info("Computing " + test_name + " Embeddings and Writing Scores")
    populate_embeddings(test_data, all_sentences, all_definitions, model, batch_size)
    scores_dict = compute_test_metrics(test_data)
    out_dir = os.path.join(results_dir, loss_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, test_name + '_results.csv')
    write_scores(out_file, scores_dict)
    logging.info("Done Writing Scores for " + test_name + " ....")
