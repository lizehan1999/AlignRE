import json
import os
import numpy as np
import data_process
import torch
import random
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import AlignRE
from transformers import get_linear_schedule_with_warmup
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-6)
parser.add_argument("--num_negsample", type=int, default=7)
parser.add_argument("--warm_up", type=float, default=0.1)
parser.add_argument("--dataset_path", type=str, default='data')
parser.add_argument("--dataset", type=str, default='fewrel', choices=['fewrel', 'wikizsl'])
parser.add_argument("--unseen", type=int, default=15)
parser.add_argument("--rel_seed", type=str, default=0)
parser.add_argument("--visible_device", type=str, default='0')
parser.add_argument("--pretrained_model", type=str, default='./BERT_MODELS/bert-base-uncased')
parser.add_argument("--prototype_model", type=str, default='./BERT_MODELS/stsb-bert-base')
parser.add_argument("--prototype_model_name", type=str, default='stsb-bert-base')
args = parser.parse_args()

args.ckpt_save_path = f'ckpt/{args.dataset}_{args.unseen}_unseen_{str(args.rel_seed)}'

args.dataset_file = os.path.join(args.dataset_path, f'{args.dataset}_dataset.json')
args.relation_description_file = os.path.join(args.dataset_path, args.dataset + "_property.json")
args.rel2id_file = os.path.join(args.dataset_path, f'{args.dataset}_rel2id.json')

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

if __name__ == "__main__":
    with open(args.rel2id_file, 'r', encoding='utf-8') as f:
        relation2idx = json.load(f)
        relation2idx = relation2idx[str(args.unseen)][args.rel_seed]
        train_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['test']
        train_idx2relation, test_idx2relation, = dict((v, k) for k, v in train_relation2idx.items()), \
            dict((v, k) for k, v in test_relation2idx.items())

    train_label, test_label = list(train_relation2idx.keys()), list(test_relation2idx.keys())

    with open(args.relation_description_file, 'r', encoding='utf-8') as rd:
        relation_desc = json.load(rd)
        train_desc = [i for i in relation_desc if i['relation'] in train_label]
        test_desc = [i for i in relation_desc if i['relation'] in test_label]
    with open(args.dataset_file, 'r', encoding='utf-8') as d:
        raw_data = json.load(d)
        train_data = [i for i in raw_data if i['relation'] in train_label]
        test_data = [i for i in raw_data if i['relation'] in test_label]

    train_relation_prototype = data_process.generate_prototype(args, train_desc, 'train')
    test_relation_prototype = data_process.generate_prototype(args, test_desc, 'test')

    config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
    config.pretrained_model = args.pretrained_model
    config.margin = args.margin
    model = AlignRE.from_pretrained(args.pretrained_model, config=config)
    model = model.cuda()

    trainset = data_process.AlignReDataset(args, 'train', train_data, train_relation_prototype, train_relation2idx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=data_process.create_mini_batch,
                             shuffle=True)

    testset = data_process.AlignReDataset(args, 'test', test_data, test_relation_prototype, test_relation2idx)
    testloader = DataLoader(testset, batch_size=10 * args.unseen,
                            collate_fn=data_process.create_mini_batch, shuffle=False)

    # test data
    test_y_attr, test_y, test_y_e1, test_y_e2 = [], [], [], []

    for i, test in enumerate(test_data):
        label = int(test_relation2idx[test['relation']])
        test_y.append(label)
    for i in test_label:
        test_y_attr.append(test_relation_prototype[i])
    test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
        test_y_e2)

    # optimizer and scheduler
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(trainset) * args.epochs // args.batch_size
    warmup_steps = num_training_steps * args.warm_up
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    best_pt, best_rt, best_f1t = 0.0, 0.0, 0.0
    test_pt, test_rt, test_f1t = 0.0, 0.0, 0.0

    # train
    for epoch in range(args.epochs):
        print(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
        running_loss = 0.0
        out_sentence_embs = None
        e1_hs = None
        e2_hs = None
        train_y = None
        for step, data in enumerate(trainloader):
            input_ids, attention_mask, token_type_ids, mark_head_mask, mark_tail_mask, mark_relation_mask, relation_emb, labels_ids = [
                t.cuda() for t in data]

            optimizer.zero_grad()

            outputs, relation_mask, e1_mask, e2_mask = model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             token_type_ids=token_type_ids,
                                                             mark_head_mask=mark_head_mask,
                                                             mark_tail_mask=mark_tail_mask,
                                                             mark_relation_mask=mark_relation_mask,
                                                             input_relation_emb=relation_emb,
                                                             labels=labels_ids,
                                                             num_neg_sample=args.num_negsample
                                                             )

            loss = outputs.sum()
            loss = loss / args.batch_size

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if step % 100 == 0:
                print(f'[step {step}]' + '=' * (step // 100))
                print('running_loss:{}'.format(running_loss / (step + 1)))

        print('============== EVALUATION ON Test DATA ==============')

        preds, e1_hs, e2_hs = extract_relation_emb(model, testloader)
        test_pt, test_rt, test_f1t = evaluate(preds.cpu(), e1_hs.cpu(), e2_hs.cpu(), test_y_attr, test_y)

        if test_f1t > best_f1t:
            best_pt, best_rt, best_f1t = test_pt, test_rt, test_f1t
            torch.save(model.state_dict(), args.ckpt_save_path + f'_f1_{test_f1t}')
        print(f'[test] precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')
        print("* " * 20)
    print(f'[test] final precision: {best_pt:.4f}, recall: {best_rt:.4f}, f1 score: {best_f1t:.4f}')
