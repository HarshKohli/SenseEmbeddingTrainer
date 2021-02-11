# SenseEmbeddingTrainer
Optimization Experiments for Word Sense Disambiguation

# Setting Up

## Download Datasets, Saved Checkpoints & Results

### Download train and eval datasets - raw as well as preprocessed from :-

https://drive.google.com/file/d/1SnOVlBrpCWFCwqlB4Cw6qOpQ07L_RF9v/view?usp=sharing

Unzip the contents and copy to the datasets/ directory

### Saved models for the various experiments can be downloaded from :-

https://drive.google.com/drive/folders/1XdNDXcMDU9hhDe8tarUBzd-yA7q_AO_c?usp=sharing

Copy the contents to the model_save/ directory

### Results files can be downlaoded from :-

https://drive.google.com/drive/folders/1QohJaYBY5dGrPuzmJ7x9rFwI4nrNaVfI?usp=sharing

For each model type, the results are as follows-

all_results.csv - contains results of the various eval datasets combined as well as scores for individual pos types.

semeval_2007.csv - contains overall results on semeval2007 as well as results on various POS types.

(Results on other eval sets are included in the respective files similarly)

## Environment & Packages

Code is tested on python3.8

To install necessary requirements:-

pip install -r requirements.txt

# Model Training & Reproducing Results

The best model is found in Hypernym_plus_Triplet and the corresponding results are present in the directory of the same name in results/ (instructions and download links mentioned above)

config.yml contains various configuration elements, dataset paths, and other hyper-parameters

In order to test the best model, set eval_base = Hypernym_plus_Triplet (default) and run eval.py . Hypernym_plus_Triplet must be present in the model_save directory and results are written in results/ folder







