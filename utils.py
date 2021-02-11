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
    def __init__(self, sentence, definition, pos, label, skey):
        self.sentence = sentence
        self.definitions = [definition]
        self.pos = pos
        self.labels = [label]
        self.skeys = [skey]
        self.sentence_embeddings = None
        self.definition_embeddings = None
        self.scores = None


class TripletSample:
    def __init__(self, anchor):
        self.anchor = anchor
        self.positives = set()
        self.negatives = set()


def get_file_data(filename):
    dataset = []
    data_file = open(filename, 'r', encoding='utf8')
    for line in data_file.readlines()[1:]:
        info = line.split('\t')
        dataset.append(InputExample(texts=[info[2].strip(), info[3].strip()], label=float(info[1].strip())))
    data_file.close()
    return dataset


def get_train_dev_data(config, train_filepath):
    train_data = get_file_data(train_filepath)
    dev_data = get_file_data(os.path.join(config['eval_dir'], config['dev_file']))
    return train_data, dev_data


def get_triplet_data(config):
    dev_data = get_file_data(os.path.join(config['eval_dir'], config['dev_file']))
    train_file = open(os.path.join(config['train_dir'], config['train_triplet_file']))
    train_dataset = []
    for line in train_file.readlines():
        info = line.strip().split('\t')
        anchor, positive, negative = info[0], info[1], info[2]
        train_dataset.append(InputExample(texts=[anchor, positive, negative]))
    train_file.close()
    return train_dataset, dev_data


def populate_embeddings(test_data, all_sentences, all_definitions, model, batch_size, embedding_lookup,
                        embedding_lookup_all, config):
    sentence_embeddings = model.encode(all_sentences, batch_size=batch_size)
    definition_embeddings = model.encode(all_definitions, batch_size=batch_size)
    def_ptr = 0
    for sentence_embed, test_key in zip(sentence_embeddings, test_data):
        test_datum = test_data[test_key]
        def_embeds = []
        for definition in test_datum.definitions:
            if config['eval_strategy'] == 'Centroid':
                if definition in embedding_lookup:
                    def_embeds.append(embedding_lookup[definition])
                else:
                    def_embeds.append(definition_embeddings[def_ptr])
            if config['eval_strategy'] == 'MaxSim':
                if definition in embedding_lookup_all:
                    all_embeds = embedding_lookup_all[definition]
                    similarities = cosine_similarity([sentence_embed], all_embeds)
                    max_sim_index = np.argmax(similarities)
                    def_embeds.append(all_embeds[max_sim_index])
                else:
                    def_embeds.append(definition_embeddings[def_ptr])
            else:
                def_embeds.append(definition_embeddings[def_ptr])
            def_ptr = def_ptr + 1
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
        target_id, sentence, definition, label, pos, skey = info[0], info[2], info[3], info[1], info[5], info[4]
        all_definitions.append(definition)
        if broadcast_sentences:
            all_sentences.append(sentence)
        if target_id in test_data:
            test_data[target_id].definitions.append(definition)
            test_data[target_id].labels.append(label)
            test_data[target_id].skeys.append(skey)
        else:
            if not broadcast_sentences:
                all_sentences.append(sentence)
            test_data[target_id] = TestSample(sentence, definition, pos, label, skey)
    test_file.close()
    return test_data, all_sentences, all_definitions


def create_hypernynm_gloss_data(config):
    train_files = config['train_raw_files']
    gloss_data, hyp_data = [], []
    oversample_ratio = config['oversample_ratio']
    triplet_data = {}
    for file in train_files:
        train_file = open(os.path.join(config['train_raw_dir'], file), 'r', encoding='utf8')
        for line in train_file.readlines()[1:]:
            info = line.strip().split('\t')
            target_id, label, sentence, gloss, sense_key = info[0], info[1], info[2], info[3], info[4]
            gloss_data.append([target_id, label, sentence, gloss, sense_key])
            if sentence in triplet_data:
                triplet_obj = triplet_data[sentence]
            else:
                triplet_obj = TripletSample(sentence)
                triplet_data[sentence] = triplet_obj
            if label == '1':
                triplet_obj.positives.add(gloss)
            else:
                triplet_obj.negatives.add(gloss)
            if label == '1':
                for _ in range(oversample_ratio - 1):
                    gloss_data.append([target_id, label, sentence, gloss, sense_key])
            try:
                wn_synset = wn.synset_from_sense_key(sense_key)
                if len(wn_synset.hypernyms()) > 0:
                    for hyp in wn_synset.hypernyms():
                        defs = hyp.definition()
                        word = hyp.lemmas()[0].name().replace('_', ' ')
                        new_gloss = word + ' : ' + defs
                        hyp_data.append([target_id, label, sentence, new_gloss, sense_key])
                        if label == '1':
                            for _ in range(oversample_ratio - 1):
                                hyp_data.append([target_id, label, sentence, new_gloss, sense_key])
            except:
                continue
        train_file.close()
    return gloss_data, hyp_data, triplet_data


def write_train_file(all_data, filepath):
    random.shuffle(all_data)
    outfile = open(filepath, 'w', encoding='utf8')
    outfile.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
    for sample in all_data:
        outfile.write(sample[0] + '\t' + sample[1] + '\t' + sample[2] + '\t' + sample[3] + '\t' + sample[4] + '\n')
    outfile.close()


def write_triplet_train_file(triplet_data, filepath):
    outfile = open(filepath, 'w', encoding='utf8')
    for _, triplet_obj in triplet_data.items():
        anchor = triplet_obj.anchor
        positives = triplet_obj.positives
        negatives = triplet_obj.negatives
        for pos in positives:
            for neg in negatives:
                outfile.write(anchor + '\t' + pos + '\t' + neg + '\n')
    outfile.close()


def get_sense_samples(train_file):
    file = open(train_file, 'r', encoding='utf8')
    def_sentences_map = {}
    for line in file.readlines()[1:]:
        info = line.strip().split('\t')
        label, sentence, gloss = info[1], info[2], info[3]
        if label == '1':
            if gloss not in def_sentences_map:
                def_sentences_map[gloss] = set()
            def_sentences_map[gloss].add(sentence)
            def_sentences_map[gloss].add(gloss)
    return def_sentences_map


def compute_sense_clusters(def_sentences_map, batch_size, embedding_lookup, embedding_lookup_all, model):
    all_sentences = []
    for _, sentences in def_sentences_map.items():
        all_sentences.extend(sentences)
    sentence_embeddings = model.encode(all_sentences, batch_size=batch_size)
    sentence_ptr = 0
    for definition, sentences in def_sentences_map.items():
        num_defs = len(sentences)
        embeddings = sentence_embeddings[sentence_ptr: sentence_ptr + num_defs]
        def_embed = np.mean(embeddings, axis=0)
        embedding_lookup[definition] = def_embed
        embedding_lookup_all[definition] = embeddings
        sentence_ptr = sentence_ptr + num_defs


def compute_test_metrics(test_data, do_cosine, semcor_keys = None):
    correct, tot = 0, 0
    vbp, vbn, nnp, nnn, adjp, adjn, advp, advn = 0, 0, 0, 0, 0, 0, 0, 0
    in_semcor_correct, in_semcor_wrong, out_semcor_correct, out_semcor_wrong = 0, 0, 0, 0
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
            if semcor_keys is not None:
                if sample.skeys[max_sim_index] in semcor_keys:
                    in_semcor_correct = in_semcor_correct + 1
                else:
                    out_semcor_correct = out_semcor_correct + 1
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
            ans_index = sample.labels.index('1')
            if semcor_keys is not None:
                if sample.skeys[ans_index] in semcor_keys:
                    in_semcor_wrong = in_semcor_wrong + 1
                else:
                    out_semcor_wrong = out_semcor_wrong + 1
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
    if semcor_keys is not None:
        print(in_semcor_correct)
        print(in_semcor_wrong)
        print(out_semcor_correct)
        print(out_semcor_wrong)

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
        batch = []
        for a, b in zip(sbatch, dbatch):
            batch.append([a, b])
        scores_raw = model.predict(batch)
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
