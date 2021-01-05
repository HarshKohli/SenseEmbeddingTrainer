# Author: Harsh Kohli
# Date created: 1/4/2021

import os
from sentence_transformers import InputExample, losses


def get_file_data(filename):
    dataset = []
    data_file = open(filename, 'r', encoding='utf8')
    for line in data_file.readlines()[1:]:
        info = line.split('\t')
        dataset.append(InputExample(texts=[info[2].strip(), info[3].strip()], label=float(info[1].strip())))
    data_file.close()
    return dataset


def get_train_dev_data(config):
    train_data = get_file_data(os.path.join(config['train_dir'], config['train_flat_file']))
    dev_data = get_file_data(os.path.join(config['eval_dir'], config['dev_file']))
    return train_data, dev_data


def get_loss(loss_type, model):
    if loss_type == 'BatchAllTripletLoss':
        return losses.BatchAllTripletLoss(model=model)

    if loss_type == 'BatchHardSoftMarginTripletLoss':
        return losses.BatchHardSoftMarginTripletLoss(model=model)

    if loss_type == 'BatchHardTripletLoss':
        return losses.BatchHardTripletLoss(model=model)

    if loss_type == 'BatchSemiHardTripletLoss':
        return losses.BatchSemiHardTripletLoss(model=model)

    if loss_type == 'ContrastiveLoss':
        return losses.ContrastiveLoss(model=model)

    if loss_type == 'CosineSimilarityLoss':
        return losses.CosineSimilarityLoss(model=model)

    if loss_type == 'MegaBatchMarginLoss':
        return losses.MegaBatchMarginLoss(model=model)

    if loss_type == 'MultipleNegativesRankingLoss':
        return losses.MultipleNegativesRankingLoss(model=model)

    if loss_type == 'OnlineContrastiveLoss':
        return losses.OnlineContrastiveLoss(model=model)

    if loss_type == 'TripletLoss':
        return losses.TripletLoss(model=model)

    raise ValueError('Invalid loss type')
