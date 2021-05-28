# SenseEmbeddingTrainer
Optimization Experiments for Word Sense Disambiguation

# Setting Up

## Download Datasets, Saved Checkpoints & Results

### Download train and eval datasets - raw as well as preprocessed from :-

https://drive.google.com/file/d/14yQliqnOTNVW5xZqWJxZALz3gSXnBitX/view?usp=sharing

Unzip the contents and copy to the datasets/ directory

### Saved models for the various experiments can be downloaded from :-

https://drive.google.com/drive/folders/1wPocJvOQD3ip95Yzie-MSsnnScOMC9r7?usp=sharing

Copy the contents to the model_save/ directory

### Results files can be downloaded from :-

https://drive.google.com/drive/folders/1d2ozj253Mr2jaRFIk78c6xHE8Y8uTQ_a?usp=sharing

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

## Retraining for the best model

1. set transfer_learn: False and use_hypernym: True in config . Then run main.py (might take over 24 hours to train, depending on hardware)

2. set transfer_learn: True, use_hypernym: False and transfer_learn_base: Hypernym in config. Run train_triplet_loss.py (again, might take over 1 day)

Run eval.py to verify results

The pretrained Hypernym model is available, so you can directly skip to step 2. described above.

## Testing other configurations

train_multi_task.py to pre-train on multi-task dataset

Available loss types (can be configured using the loss_type flag in config.yml) are - [BatchAllTripletLoss, BatchHardSoftMarginTripletLoss, BatchHardTripletLoss, BatchSemiHardTripletLoss, ContrastiveLoss, CosineSimilarityLoss, MegaBatchMarginLoss, MultipleNegativesRankingLoss, OnlineContrastiveLoss]

For fine-tuning on any previously saved model, set transfer_learn: True and transfer_learn_base must point to the pretrained model directory in models/









