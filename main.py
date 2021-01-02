import yaml
from sentence_transformers import LoggingHandler, SentenceTransformer
import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_HOME"] = config['base_model_dir']
model_name = config['base_model']
train_batch_size = config['batch_size']

model = SentenceTransformer(model_name)

logging.info("Read SemCor train dataset")
