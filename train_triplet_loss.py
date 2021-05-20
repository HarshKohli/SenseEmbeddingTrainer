# Author: Harsh Kohli
# Date created: 1/24/2021

import yaml
import logging
import os
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset, evaluation, losses
from utils import get_triplet_data

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']

if config['transfer_learn']:
    model = SentenceTransformer(os.path.join(config['saved_model_dir'], config['transfer_learn_base']))
else:
    model = SentenceTransformer(config['base_model'])

loss_type = 'TripletLoss'
model_dir = os.path.join(config['saved_model_dir'], config['checkpoint_path'])

logging.info("Processing Data ...")
train_samples, dev_samples = get_triplet_data(config)
logging.info("Done Processing Data ...")

batch_size = config['batch_size']
num_epochs = config['num_epochs']

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
train_loss = losses.TripletLoss(model=model)
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
