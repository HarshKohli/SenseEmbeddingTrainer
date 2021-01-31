# Author: Harsh Kohli
# Date created: 1/1/2021

import yaml
import logging
import os
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset, evaluation
from utils import get_train_dev_data, get_loss

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']

model_dir = os.path.join(config['saved_model_dir'], config['checkpoint_path'])
if config['transfer_learn']:
    model = SentenceTransformer(os.path.join(config['saved_model_dir'], config['transfer_learn_base']))
else:
    model = SentenceTransformer(config['base_model'])

if config['use_hypernym']:
    train_file = os.path.join(config['train_dir'], config['train_hyp_file'])
else:
    train_file = os.path.join(config['train_dir'], config['train_flat_file'])

logging.info("Processing Data ...")
train_samples, dev_samples = get_train_dev_data(config, train_file)
logging.info("Done Processing Data ...")

batch_size = config['batch_size']
num_epochs = config['num_epochs']

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
train_loss = get_loss(config['loss_type'], model)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples)

warmup_steps = math.ceil(len(train_dataset) * num_epochs / batch_size * float(config['warmup_ratio']))
logging.info("Warmup-steps: {}".format(warmup_steps))

logging.info("Starting training ...")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=int(config['eval_steps']),
          warmup_steps=warmup_steps,
          output_path=model_dir)
