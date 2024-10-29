import torch
import random
from transformers import BertModel, BertPreTrainedModel
from utils import calculate_similarity


def extract_mask(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    return extended_e_mask.float()


class AlignRE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.margin = torch.tensor(config.margin)
        self.bert = BertModel(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mark_head_mask=None,
            mark_tail_mask=None,
            mark_relation_mask=None,
            input_relation_emb=None,
            labels=None,
            num_neg_sample=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        e1_mask = extract_mask(sequence_output, mark_head_mask)
        e2_mask = extract_mask(sequence_output, mark_tail_mask)
        relation_mask = extract_mask(sequence_output, mark_relation_mask)

        e1_mask = torch.tanh(e1_mask)
        e2_mask = torch.tanh(e2_mask)
        relation_mask = torch.tanh(relation_mask)

        outputs = (outputs,)
        if labels is not None:
            margin = self.margin.cuda()
            loss = torch.tensor(0.).cuda()
            zeros = torch.tensor(0.).cuda()
            for a, b in enumerate(zip(relation_mask, e1_mask, e2_mask)):
                max_val = torch.tensor(0.).cuda()
                matched_sentence_pair = input_relation_emb[a]
                pos_s = calculate_similarity(matched_sentence_pair, (b[0] + b[1] + b[2]) / 3).cuda()
                pos = pos_s

                if num_neg_sample > len(input_relation_emb):
                    break
                else:
                    rand = random.sample(range(len(input_relation_emb)), num_neg_sample)
                neg_relation_emb = torch.stack([input_relation_emb[i] for i in rand])

                for i, j in enumerate(zip(neg_relation_emb)):
                    tmp_s = calculate_similarity((b[0] + b[1] + b[2]) / 3, j[0]).cuda()
                    tmp = tmp_s
                    if tmp > max_val:
                        if (matched_sentence_pair == j[0]).all():
                            continue
                        else:
                            max_val = tmp

                neg = max_val.cuda()
                loss += torch.max(zeros, neg - pos + margin)
            outputs = loss
        return outputs, relation_mask, e1_mask, e2_mask
