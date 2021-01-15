# Author: Harsh Kohli
# Date created: 1/4/2021

import os
import random
import numpy as np
from nltk.corpus import wordnet as wn
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import InputExample, losses


class TestSample:
    def __init__(self, sentence, definition, pos, label):
        self.sentence = sentence
        self.definitions = [definition]
        self.pos = pos
        self.labels = [label]
        self.sentence_embeddings = None
        self.definition_embeddings = None
        self.scores = None


def get_file_data(filename):
    dataset = []
    data_file = open(filename, 'r', encoding='utf8')
    for line in data_file.readlines()[1:]:
        info = line.split('\t')
        dataset.append(InputExample(texts=[info[2].strip(), info[3].strip()], label=float(info[1].strip())))
    data_file.close()
    return dataset


def get_train_dev_data(config, use_hypernym):
    if use_hypernym:
        train_data = get_file_data(os.path.join(config['train_dir'], config['train_hyp_file']))
    else:
        train_data = get_file_data(os.path.join(config['train_dir'], config['train_flat_file']))
    dev_data = get_file_data(os.path.join(config['eval_dir'], config['dev_file']))
    return train_data, dev_data


def populate_embeddings(test_data, all_sentences, all_definitions, model, batch_size):
    sentence_embeddings = model.encode(all_sentences, batch_size=batch_size)
    definition_embeddings = model.encode(all_definitions, batch_size=batch_size)
    def_ptr = 0
    for sentence_embed, test_key in zip(sentence_embeddings, test_data):
        test_datum = test_data[test_key]
        num_defs = len(test_datum.definitions)
        def_embeds = definition_embeddings[def_ptr: def_ptr + num_defs]
        def_ptr = def_ptr + num_defs
        test_datum.sentence_embeddings = sentence_embed
        test_datum.definition_embeddings = def_embeds


def populate_scores(test_data, scores):
    def_ptr = 0
    for test_key, test_datum in test_data.items():
        test_datum = test_data[test_key]
        num_defs = len(test_datum.definitions)
        batch_scores = scores[def_ptr: def_ptr + num_defs]
        def_ptr = def_ptr + num_defs
        test_datum.scores = batch_scores


def get_test_data(filename, broadcast_sentences):
    test_file = open(filename, 'r', encoding='utf8')
    test_data = {}
    all_sentences, all_definitions = [], []
    for line in test_file.readlines()[1:]:
        info = line.strip().split('\t')
        target_id, sentence, definition, label, pos = info[0], info[2], info[3], info[1], info[5]
        all_definitions.append(definition)
        if broadcast_sentences:
            all_sentences.append(sentence)
        if target_id in test_data:
            test_data[target_id].definitions.append(definition)
            test_data[target_id].labels.append(label)
        else:
            if not broadcast_sentences:
                all_sentences.append(sentence)
            test_data[target_id] = TestSample(sentence, definition, pos, label)
    test_file.close()
    return test_data, all_sentences, all_definitions


def create_hypernynm_data(config):
    train_file = open(os.path.join(config['train_dir'], config['train_flat_file']), 'r', encoding='utf8')

    all_data = []
    for line in train_file.readlines()[1:]:
        info = line.strip().split('\t')
        target_id, label, sentence, gloss, sense_key = info[0], info[1], info[2], info[3], info[4]
        all_data.append([target_id, label, sentence, gloss, sense_key])
        if label == '1':
            for _ in range(5):
                all_data.append([target_id, label, sentence, gloss, sense_key])
        try:
            wn_synset = wn.synset_from_sense_key(sense_key)
            if len(wn_synset.hypernyms()) > 0:
                for hyp in wn_synset.hypernyms():
                    defs = hyp.definition()
                    word = hyp.lemmas()[0].name().replace('_', ' ')
                    new_gloss = word + ' : ' + defs
                    if label == '1':
                        new_label = '2'
                    else:
                        new_label = '0'
                    all_data.append([target_id, new_label, sentence, new_gloss, sense_key])

                    if label == '1':
                        for _ in range(7):
                            all_data.append([target_id, new_label, sentence, new_gloss, sense_key])
        except:
            continue

    random.shuffle(all_data)
    train_file.close()
    return all_data


def write_hypernym_train_file(all_data, config):
    outfile = open(os.path.join(config['train_dir'], config['train_hyp_file']), 'w', encoding='utf8')
    outfile.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
    for sample in all_data:
        outfile.write(sample[0] + '\t' + sample[1] + '\t' + sample[2] + '\t' + sample[3] + '\t' + sample[4] + '\n')
    outfile.close()


def compute_test_metrics(test_data, do_cosine):
    correct, tot = 0, 0
    vbp, vbn, nnp, nnn, adjp, adjn, advp, advn = 0, 0, 0, 0, 0, 0, 0, 0
    for target_id, sample in test_data.items():
        if do_cosine is True:
            sentence_embeddings = sample.sentence_embeddings
            definition_embeddings = sample.definition_embeddings
            similarities = cosine_similarity([sentence_embeddings], definition_embeddings)
        else:
            similarities = sample.scores
        max_sim_index = np.argmax(similarities)
        pos = sample.pos
        if sample.labels[max_sim_index] == '1':
            correct = correct + 1
            if pos == 'NOUN':
                nnp = nnp + 1
            elif pos == 'VERB':
                vbp = vbp + 1
            elif pos == 'ADJ':
                adjp = adjp + 1
            elif pos == 'ADV':
                advp = advp + 1
            else:
                raise ValueError('Invalid POS type')
        else:
            if pos == 'NOUN':
                nnn = nnn + 1
            elif pos == 'VERB':
                vbn = vbn + 1
            elif pos == 'ADJ':
                adjn = adjn + 1
            elif pos == 'ADV':
                advn = advn + 1
            else:
                raise ValueError('Invalid POS type')
        tot = tot + 1

    scores_dict = {}
    if nnn + nnp > 0:
        scores_dict['NOUN'] = nnp / (nnp + nnn)
    else:
        scores_dict['NOUN'] = 0
    if vbp + vbn > 0:
        scores_dict['VERB'] = vbp / (vbp + vbn)
    else:
        scores_dict['VERB'] = 0
    if adjp + adjn > 0:
        scores_dict['ADJ'] = adjp / (adjp + adjn)
    else:
        scores_dict['ADJ'] = 0
    if advp + advn > 0:
        scores_dict['ADV'] = advp / (advp + advn)
    else:
        scores_dict['ADV'] = 0

    scores_dict['TOTAL'] = correct / tot
    return scores_dict


def write_scores(filename, scores_dict):
    results_file = open(filename, 'w', encoding='utf8')
    for type, score in scores_dict.items():
        results_file.write(type + '\t' + str(score) + '\n')
    results_file.close()


def get_crossencoder_scores(all_sentences, all_definitions, batch_size, model):
    sentence_batches = [all_sentences[i * batch_size:(i + 1) * batch_size] for i in
                        range((len(all_sentences) + batch_size - 1) // batch_size)]

    definition_batches = [all_definitions[i * batch_size:(i + 1) * batch_size] for i in
                          range((len(all_definitions) + batch_size - 1) // batch_size)]

    scores = []
    for sbatch, dbatch in zip(sentence_batches, definition_batches):
        scores_raw = model.predict([sbatch, dbatch])
        scores_normalized = softmax(scores_raw, axis=1)
        scores_normalized = [score[1] for score in scores_normalized]
        scores.extend(scores_normalized)
    return scores


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

    raise ValueError('Invalid loss type')
