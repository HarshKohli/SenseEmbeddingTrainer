# Author: Harsh Kohli
# Date created: 1/24/2021

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
model = SentenceTransformer(config['base_model'])

model_dir = os.path.join(config['saved_model_dir'], config['checkpoint_path'])
gloss_train_file = os.path.join(config['train_dir'], config['train_hyp_file'])
hyp_train_file = os.path.join(config['train_dir'], config['train_flat_file'])

logging.info("Processing Data ...")
gloss_train_samples, dev_samples = get_train_dev_data(config, gloss_train_file)
hyp_train_samples, _ = get_train_dev_data(config, hyp_train_file)
logging.info("Done Processing Data ...")

batch_size = config['batch_size']
num_epochs = config['num_epochs']

gloss_train_dataset = SentencesDataset(gloss_train_samples, model)
gloss_train_dataloader = DataLoader(gloss_train_dataset, shuffle=True, batch_size=batch_size)
gloss_train_loss = get_loss(config['loss_type'], model)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples)

hyp_train_dataset = SentencesDataset(hyp_train_samples, model)
hyp_train_dataloader = DataLoader(hyp_train_dataset, shuffle=True, batch_size=batch_size)
hyp_train_loss = get_loss(config['loss_type'], model)

warmup_steps = math.ceil(
    (len(gloss_train_dataset) + len(hyp_train_dataset)) * num_epochs / batch_size * float(config['warmup_ratio']))
logging.info("Warmup-steps: {}".format(warmup_steps))

logging.info("Starting training ...")
model.fit(train_objectives=[(gloss_train_dataloader, gloss_train_loss), (hyp_train_dataloader, hyp_train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=int(config['eval_steps']),
          warmup_steps=warmup_steps,
          output_path=model_dir)
