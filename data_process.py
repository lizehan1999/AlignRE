import json
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from utils import weight_mean


def sentence_bert_prototype(args, description_sentences):
    emb_dict = {}
    encoder = SentenceTransformer(args.prototype_model)
    print("=============ReBuild=============")

    description_sentences_text = [i[1] for i in description_sentences]
    label_name_sentences_text = [i[2] for i in description_sentences]

    aliases_sentences_text_0, aliases_sentences_text_1, aliases_sentences_text_2 = [], [], []
    for i in description_sentences:
        random_id = random.sample(range(0, len(i[3])), 3)
        aliases_sentences_text_0.append(i[3][random_id[0]])
        aliases_sentences_text_1.append(i[3][random_id[1]])
        aliases_sentences_text_2.append(i[3][random_id[2]])

    description_sentence_embeddings = encoder.encode(description_sentences_text)
    label_name_sentence_embeddings = encoder.encode(label_name_sentences_text)
    aliases_name_sentence_embeddings_0 = encoder.encode(aliases_sentences_text_0)
    aliases_name_sentence_embeddings_1 = encoder.encode(aliases_sentences_text_1)
    aliases_name_sentence_embeddings_2 = encoder.encode(aliases_sentences_text_2)

    description_sentence_embeddings = [embedding for embedding in description_sentence_embeddings]
    label_name_sentence_embeddings = [embedding for embedding in label_name_sentence_embeddings]
    aliases_name_sentence_embeddings_0 = [embedding for embedding in aliases_name_sentence_embeddings_0]
    aliases_name_sentence_embeddings_1 = [embedding for embedding in aliases_name_sentence_embeddings_1]
    aliases_name_sentence_embeddings_2 = [embedding for embedding in aliases_name_sentence_embeddings_2]
    for i, (emb_d, emb_l, emb_a0, emb_a1, emb_a2) in enumerate(
            zip(description_sentence_embeddings, label_name_sentence_embeddings, aliases_name_sentence_embeddings_0,
                aliases_name_sentence_embeddings_1, aliases_name_sentence_embeddings_2)):
        emb_dict[description_sentences[i][0]] = weight_mean([emb_d, emb_l, emb_a0, emb_a1, emb_a2]).tolist()

    return emb_dict


def get_prototype_emb(args, description_sentences, mode, model_name="stsb-bert-base"):
    description_sentence_embeddings = sentence_bert_prototype(args, description_sentences)

    with open(args.dataset_path + "/" + args.dataset + "_" + mode + "_" + model_name + ".json", 'w') as f:
        json.dump(description_sentence_embeddings, f)

    return description_sentence_embeddings


def generate_prototype(args, description_file, mode):
    model_name = args.prototype_model_name
    description_sentences = [
        (single_data['relation'], single_data['description'], single_data['label_name'], single_data['aliases']) for
        single_data in description_file]

    prototype_file = args.dataset_path + "/" + args.dataset + "_" + mode + "_" + model_name + ".json"
    if not os.path.exists(prototype_file):
        get_prototype_emb(args, description_sentences, mode, model_name)

    rel2vec = {}
    with open(prototype_file, 'r') as f:
        prototype_embs = json.load(f)
    for d in description_sentences:
        rel2vec[d[0]] = np.array(prototype_embs[d[0]]).astype('float32')

    return rel2vec


def mark_entity_and_mask(mask_idx, sent_len):
    mark_head_mask = np.array([0] * sent_len)
    mark_tail_mask = np.array([0] * sent_len)
    mark_relation_mask = np.array([0] * sent_len)
    mark_head_mask[mask_idx[0]] = 1
    mark_tail_mask[mask_idx[1]] = 1
    mark_relation_mask[mask_idx[2]] = 1
    return torch.tensor(mark_head_mask, dtype=torch.long), \
        torch.tensor(mark_tail_mask, dtype=torch.long), \
        torch.tensor(mark_relation_mask, dtype=torch.long),


def create_mini_batch(samples):
    input_ids = [s[0] for s in samples]
    attention_mask = [s[1] for s in samples]
    token_type_ids = [s[2] for s in samples]

    mark_head_mask = [s[3] for s in samples]
    mark_tail_mask = [s[4] for s in samples]
    mark_relation_mask = [s[5] for s in samples]

    relation_emb = [s[6] for s in samples]

    if samples[0][7] is not None:
        labels_ids = torch.stack([s[7] for s in samples])
    else:
        labels_ids = None

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)

    mark_head_mask = pad_sequence(mark_head_mask, batch_first=True)
    mark_tail_mask = pad_sequence(mark_tail_mask, batch_first=True)
    mark_relation_mask = pad_sequence(mark_relation_mask, batch_first=True)

    relation_emb = torch.tensor(relation_emb)

    return input_ids, attention_mask, token_type_ids, mark_head_mask, mark_tail_mask, mark_relation_mask, relation_emb, labels_ids


class AlignReDataset(Dataset):
    def __init__(self, args, mode, data, relation_prototype, relation2idx):
        assert mode in ['train', 'test']
        self.args = args
        self.mode = mode
        self.data = data
        self.relation_prototype = relation_prototype
        self.relation2idx = relation2idx
        self.len = len(data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model)

        self.head_mark_ids = 1001
        self.tail_mark_ids = 1030

    def __getitem__(self, idx):
        single_data = self.data[idx]
        pos1 = single_data['h']['pos'][0]
        pos1_end = single_data['h']['pos'][1]
        pos2 = single_data['t']['pos'][0]
        pos2_end = single_data['t']['pos'][1]
        words = single_data['token']

        if pos1 < pos2:
            new_words = words[:pos1] + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:pos2] \
                        + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:]

        else:
            new_words = words[:pos2] + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:pos1] \
                        + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:]

        sentence = " ".join(new_words)

        prompt = "The relation between [MASK] {} and [MASK] {} is \"[MASK]\"".format(
            " ".join(words[pos1:pos1_end]),
            " ".join(words[pos2:pos2_end]))

        tokens_info = self.tokenizer(sentence, prompt)

        input_ids = tokens_info['input_ids']
        attention_mask = torch.tensor(tokens_info['attention_mask'])
        token_type_ids = torch.tensor(tokens_info['token_type_ids'])

        if pos2 == 0:
            input_ids[1] = self.tail_mark_ids
        elif pos1 == 0:
            input_ids[1] = self.head_mark_ids

        mask_idx = [index for index, value in enumerate(input_ids) if value == 103]

        mark_head_mask, mark_tail_mask, mark_relation_mask = \
            mark_entity_and_mask(
                mask_idx,
                len(input_ids)
            )
        relation_prototype = self.relation_prototype[single_data['relation']]
        input_ids = torch.tensor(input_ids)
        label_idx_tensor = None
        if self.mode == 'train':
            label_idx = int(self.relation2idx[single_data['relation']])
            label_idx_tensor = torch.tensor(label_idx)
        elif self.mode == 'test':
            label_idx_tensor = None

        return (
            input_ids, attention_mask, token_type_ids, mark_head_mask,
            mark_tail_mask, mark_relation_mask, relation_prototype, label_idx_tensor)

    def __len__(self):
        return self.len
