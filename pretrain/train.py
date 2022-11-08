from transformers import AdamW, BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
import numpy as np
import os
import pickle
import utils
import torch
from model import HiTrans_GRU
from sklearn.metrics import f1_score, precision_score, recall_score
import logging


class DataFrame(Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.act_seq = data['act_seq']
        self.sat = data['sat']
    
    def __getitem__(self, index):
        return self.input_ids[index], self.act_seq[index], self.sat[index]
    
    def __len__(self):
        return len(self.input_ids)


def collate_fn(data):
    input_ids, act_seq, sat = zip(*data)
    batch_size = len(input_ids)
    act_seq = [torch.tensor(item).long() for item in act_seq]
    act_seq = pad_sequence(act_seq, batch_first=True, padding_value=-1)
    dialog_len = len(act_seq[0])

    pad_input_ids = []
    for dialog in input_ids:
        x = dialog + [[101,102]] * (dialog_len - len(dialog))
        pad_input_ids.append(x)
    input_ids = [torch.tensor(item).long() for dialog in pad_input_ids for item in dialog]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.view(batch_size, dialog_len, -1)

    return {'input_ids':  input_ids,
            'act_seq': act_seq,
            'sat': torch.tensor(sat).long()
            }


def train(args):
    print('[TRAIN]')

    data_name = args.data.replace('\r', '')
    model_name = args.model.replace('\r', '')

    name = f'{data_name}_{model_name}'
    print('TRAIN ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.DEBUG, filename=save_path + '/log.txt', filemode='a')

    tokenizer = BertTokenizer.from_pretrained(args.bert_name, cache_dir=args.cache_dir)

    data = utils.load_data(args, tokenizer)

    model = HiTrans_GRU(args=args, vocab_size=tokenizer.vocab_size)
    optimizer = AdamW(model.parameters(), 2e-5)

    act_best_result = [0. for _ in range(4)]
    sat_best_result = [0. for _ in range(4)]

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    
    utils.set_seed(args.seed)
    batch_size = args.batch_size * max(1, len(args.device_id))

    for i in range(args.epoch_num):
        logging.info('train epoch, {}, {}'.format(i, name))
        train_loader = DataLoader(DataFrame(data['train']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        tk0 = tqdm(train_loader)
        model.train()
        epoch_loss = [0. for _ in range(2)]
        for j, batch in enumerate(tk0):
            input_ids = batch['input_ids'].to(args.device)
            act_seq = batch['act_seq'].to(args.device)
            sat = batch['sat'].to(args.device)

            act_pred, sat_pred, act_loss, sat_loss = model(input_ids=input_ids, act_seq=act_seq, sat=sat)
            
            loss = 0.1*act_loss + sat_loss

            if len(args.device_id) > 1:
                act_loss = act_loss.mean()  # mean() to average on multi-gpu parallel training
                sat_loss = sat_loss.mean()

            epoch_loss[0] += sat_loss
            epoch_loss[1] += act_loss
            loss = 0.01 * act_loss + sat_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = [x/len(tk0) for x in epoch_loss]
        logging.info(f'loss: {epoch_loss}')

        act_test_result, sat_test_result = test(model, DataFrame(data['test']), args)
        if sat_test_result[3] > sat_best_result[3]:
            model_to_save = model.module if hasattr(model,
                        'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), save_path+'/best_pretrain.pt')
        act_best_result = [max(i1, i2) for i1, i2 in zip(act_test_result, act_best_result)]
        sat_best_result = [max(i1, i2) for i1, i2 in zip(sat_test_result, sat_best_result)]
        logging.info(f'satisfaction: test_result={sat_test_result}, intent: test_result={act_test_result}')
        logging.info(f'satisfaction: best_result={sat_best_result}, intent: best_result={act_best_result}')


def test(model, test_data, args):
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    tk0 = tqdm(test_loader)

    act_prediction = []
    sat_prediction = []
    act_label = []
    sat_label = []

    model.eval()
    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        act_seq = batch['act_seq'].to(args.device)
        sat = batch['sat'].to(args.device)
        with torch.no_grad():
            act_pred, sat_pred = model(input_ids=input_ids, act_seq=act_seq)
        act_prediction.extend(act_pred.argmax(dim=-1).cpu().tolist())
        sat_prediction.extend(sat_pred.argmax(dim=-1).cpu().tolist())
        act_label.extend(act_seq.cpu().tolist())
        sat_label.extend(sat.cpu().tolist())
    
    # satisfaction evaluation
    acc = sum([int(p == l) for p, l in zip(sat_prediction, sat_label)]) / len(sat_label)
    precision = precision_score(sat_label, sat_prediction, average='macro', zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average='macro', zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average='macro', zero_division=0)
    sat_result = (acc, precision, recall, f1)

    # act evaluation
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for ap, al in zip(act_prediction, act_label):
        al = [x for x in al if x != -1]
        ap = ap[:len(al)]
        acc_list.append(sum([int(p == l) for p, l in zip(ap, al)]) / len(al))
        precision_list.append(precision_score(al, ap, average='macro', zero_division=0))
        recall_list.append(recall_score(al, ap, average='macro', zero_division=0))
        f1_list.append(f1_score(al, ap, average='macro', zero_division=0))
    acc = sum(acc_list)/len(acc_list)
    precision = sum(precision_list)/len(precision_list)
    recall = sum(recall_list)/len(recall_list)
    f1 = sum(f1_list)/len(f1_list)
    act_result = (acc, precision, recall, f1)

    return act_result, sat_result