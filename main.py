import yaml
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']
model_name = config['base_model']
train_batch_size = config['batch_size']

model = SentenceTransformer(model_name)

logging.info("Processing SemCor train dataset")

train_samples = []
train_file = open(os.path.join(config['train_dir'], config['train_flat_file']), 'r', encoding='utf8')
for line in train_file.readlines()[1:]:
    info = line.split('\t')
    train_samples.append(InputExample(texts=[info[2].strip(), info[3].strip()], label=int(info[1].strip())))

logging.info("Done Processing SemCor train dataset")
