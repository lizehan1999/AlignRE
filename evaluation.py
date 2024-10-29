import torch
import numpy as np

from utils import calculate_similarity


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def extract_relation_emb(model, testloader):
    out_sentence_embs = None
    e1_hs = None
    e2_hs = None
    model.eval()
    for data in testloader:
        input_ids, attention_mask, token_type_ids, mark_head_mask, mark_tail_mask, mark_relation_mask, relation_emb = [
            t.cuda() for t in data if t is not None]

        with torch.no_grad():
            outputs, relation_mask, e1_mask, e2_mask = model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             token_type_ids=token_type_ids,
                                                             mark_head_mask=mark_head_mask,
                                                             mark_tail_mask=mark_tail_mask,
                                                             mark_relation_mask=mark_relation_mask,
                                                             input_relation_emb=relation_emb,
                                                             )

        if out_sentence_embs is None:
            out_sentence_embs = relation_mask
            e1_hs = e1_mask
            e2_hs = e2_mask
        else:
            out_sentence_embs = torch.cat((out_sentence_embs, relation_mask))
            e1_hs = torch.cat((e1_hs, e1_mask))
            e2_hs = torch.cat((e2_hs, e2_mask))

    return out_sentence_embs, e1_hs, e2_hs


def evaluate(preds, e1_hs, e2_hs, y_attr, true_label):
    predictions = []

    for i, (p, e1, e2) in enumerate(zip(preds, e1_hs, e2_hs)):
        similarities = calculate_similarity(torch.tensor(y_attr).cuda(), (p.cuda() + e1.cuda() + e2.cuda()) / 3)
        result = similarities
        prediction = result.argmax(axis=0)
        predictions.append(prediction)

    p_macro, r_macro, f_macro = compute_macro_PRF(np.array([i.cpu() for i in predictions]), np.array(
        true_label))

    return p_macro, r_macro, f_macro
