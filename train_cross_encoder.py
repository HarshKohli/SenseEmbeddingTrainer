# Author: Harsh Kohli
# Date created: 1/15/2021

import yaml
import logging
import os
from torch.utils.data import DataLoader
import math
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from utils import get_train_dev_data, get_loss

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']

folder_name = 'CrossEncoder'
num_labels = 2
if config['use_hyperynm']:
    folder_name = folder_name + '_w_hypernym'
    num_labels = 3

model = CrossEncoder('microsoft/deberta-base', num_labels=3)

logging.info("Processing Data ...")
train_samples, dev_samples = get_train_dev_data(config)
logging.info("Done Processing Data ...")

batch_size = config['batch_size']
num_epochs = config['num_epochs']

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
train_loss = get_loss(config['loss_type'], model)
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(dev_samples)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

model_dir = os.path.join(config['saved_model_dir'], folder_name)

logging.info("Starting training ...")
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=int(config['eval_steps']),
          warmup_steps=warmup_steps,
          output_path=model_dir)
