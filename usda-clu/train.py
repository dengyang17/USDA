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
import torch.nn.functional as F
from model import Pretrained_HiTrans, USDA, USDA_A, USDA_P
from sklearn.metrics import f1_score, precision_score, recall_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure
import logging


class DataFrame(Dataset):
    def __init__(self, data, args):
        self.input_ids = data['input_ids']
        self.act_seq = data['act_seq']
        self.sat = data['sat']
        self.max_len = args.max_dia_len
    
    def __getitem__(self, index):
        return self.input_ids[index][-self.max_len:], self.act_seq[index][-self.max_len:], self.sat[index]
    
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

    name = f'{data_name}_{model_name}_{args.max_dia_len}'
    if args.pretrain_model is not None:
        name += '_pretrain'
    print('TRAIN ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.DEBUG, filename=save_path + '/log.txt', filemode='a')

    tokenizer = BertTokenizer.from_pretrained(args.bert_name, cache_dir=args.cache_dir)

    data = utils.load_data(args, tokenizer)
    act_num = len(data['act_list'])

    if args.pretrain_model is not None:
        pretrain_model = Pretrained_HiTrans(args=args, vocab_size=tokenizer.vocab_size)
        checkpoint = torch.load(args.pretrain_model)
        pretrain_model.load_state_dict(checkpoint)
        pretrained_private = pretrain_model.private
        pretrained_encoder = pretrain_model.encoder
    else:
        pretrained_private = None
        pretrained_encoder = None
    if model_name == 'USDA':
        model = USDA(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num, pretrained_private=pretrained_private, pretrained_encoder=pretrained_encoder)
    elif model_name == 'USDA_A':
        model = USDA_A(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num, pretrained_private=pretrained_private, pretrained_encoder=pretrained_encoder)
    elif model_name == 'USDA_P':
        model = USDA_P(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num, pretrained_private=pretrained_private, pretrained_encoder=pretrained_encoder)
    optimizer = AdamW(model.parameters(), 2e-5)
    
    act_best_result = [0. for _ in range(4)]
    sat_best_result = [0. for _ in range(4)]

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    
    utils.set_seed(args.seed)
    batch_size = args.batch_size * max(1, len(args.device_id))

    '''
    act_test_result, sat_test_result = test(model, DataFrame(data['test']), args)
    act_best_result = [max(i1, i2) for i1, i2 in zip(act_test_result, act_best_result)]
    sat_best_result = [max(i1, i2) for i1, i2 in zip(sat_test_result, sat_best_result)]
    logging.info(f'satisfaction: test_result={sat_test_result}, intent: test_result={act_test_result}')
    logging.info(f'satisfaction: best_result={sat_best_result}, intent: best_result={act_best_result}')
    '''

    for i in range(args.epoch_num):
        logging.info('train epoch, {}, {}'.format(i, name))
        train_loader = DataLoader(DataFrame(data['train'], args), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        tk0 = tqdm(train_loader)
        model.train()
        epoch_loss = [0. for _ in range(4)]
        for j, batch in enumerate(tk0):
            input_ids = batch['input_ids'].to(args.device)
            act_seq = batch['act_seq'].to(args.device)
            sat = batch['sat'].to(args.device)

            act_pred, sat_pred, act_loss, sat_loss = model(input_ids=input_ids, act_seq=act_seq, sat=sat)
            
            #loss = act_loss + sat_loss

            if len(args.device_id) > 1:
                #loss = loss.mean()  # mean() to average on multi-gpu parallel training
                sat_loss = sat_loss.mean()
                rec_loss = act_loss[0].mean()
                sr_loss = act_loss[1].mean()
                reg = act_loss[2].mean()
            
            epoch_loss[0] += sat_loss
            epoch_loss[1] += rec_loss
            epoch_loss[2] += sr_loss
            epoch_loss[3] += reg
            loss = sat_loss + 0.1 * (rec_loss + sr_loss + 10*reg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = [x/len(tk0) for x in epoch_loss]
        logging.info(f'loss: {epoch_loss}')

        act_test_result, sat_test_result = test(model, DataFrame(data['valid'], args), args)
        #act_best_result = [max(i1, i2) for i1, i2 in zip(act_test_result, act_best_result)]
        #sat_best_result = [max(i1, i2) for i1, i2 in zip(sat_test_result, sat_best_result)]
        if sat_test_result[-1] > sat_best_result[-1]:
            sat_best_result = sat_test_result
            act_best_result = act_test_result
            model_to_save = model.module if hasattr(model,
                        'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), save_path+'/best_pretrain.pt')
        logging.info(f'satisfaction: valid_result={sat_test_result}, intent: valid_result={act_test_result}')
        logging.info(f'satisfaction: best_valid_result={sat_best_result}, intent: best_valid_result={act_best_result}')
        sat_test_result, sat_test_result = test(model, DataFrame(data['test'], args), args)
        logging.info(f'satisfaction: test_result={sat_test_result}, intent: test_result={act_test_result}')

    # evaluation
    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_pretrain.pt')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    act_test_result, sat_test_result = test(model, DataFrame(data['test'], args), args)
    logging.info(f'satisfaction: test_result={sat_test_result}, intent: test_result={act_test_result}')


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
            act_pred, sat_pred, _, _, _ = model(input_ids=input_ids, act_seq=act_seq)
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

    '''
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
    '''
    act_result = [0, 0, 0, 0]

    return act_result, sat_result


def evaluate(args):
    print('[EVALUATE]')

    data_name = args.data.replace('\r', '')
    model_name = args.model.replace('\r', '')

    name = f'{data_name}_{model_name}_{args.max_dia_len}'
    if args.pretrain_model is not None:
        name += '_pretrain'
    print('EVALUATE ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.DEBUG, filename=save_path + '/log_eval.txt', filemode='a')

    tokenizer = BertTokenizer.from_pretrained(args.bert_name, cache_dir=args.cache_dir)

    data = utils.load_data(args, tokenizer)
    act_num = len(data['act_list'])

    if model_name == 'USDA':
        model = USDA(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num)
    elif model_name == 'USDA_A':
        model = USDA_A(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num)
    else:
        model = USDA_P(args=args, vocab_size=tokenizer.vocab_size, class_num=args.class_num)

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)

    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_pretrain.pt')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
   
    print(sum(p.numel() for p in model.parameters())) 
    test_loader = DataLoader(DataFrame(data[f'{args.eval_set}'], args), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    tk0 = tqdm(test_loader)

    act_prediction = []
    sat_prediction = []
    act_label = []
    sat_label = []

    act_res = []

    pointer_list = []
    att_a_list = []
    att_c_list = []

    model.eval()
    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        act_seq = batch['act_seq'].to(args.device)
        sat = batch['sat'].to(args.device)
        with torch.no_grad():
            act_pred, sat_pred, pointer, att_a, att_c = model(input_ids=input_ids, act_seq=act_seq)
        act_prediction.extend(act_pred.argmax(dim=-1).cpu().tolist())
        sat_prediction.extend(sat_pred.argmax(dim=-1).cpu().tolist())
        act_label.extend(act_seq.cpu().tolist())
        sat_label.extend(sat.cpu().tolist())
        act_res.extend(F.softmax(act_pred, dim=-1).cpu().tolist())

        pointer_list.extend(pointer.squeeze(-1).cpu().tolist())
        att_a_list.extend(att_a.cpu().tolist())
        att_c_list.extend(att_c.cpu().tolist())

    # satisfaction evaluation
    acc = sum([int(p == l) for p, l in zip(sat_prediction, sat_label)]) / len(sat_label)
    precision = precision_score(sat_label, sat_prediction, average='macro', zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average='macro', zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average='macro', zero_division=0)
    sat_result = (acc, precision, recall, f1)

    precision = precision_score(sat_label, sat_prediction, average=None, zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average=None, zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average=None, zero_division=0)
    sat_result_details = (precision, recall, f1)

    # save clustering representive words
    representive_words(data[f'{args.eval_set}']['input_text'], act_res, act_label, args.class_num, args.stop_path, save_path)
    
    # act evaluation
    flat_act_pred = []
    flat_act_true = []
    for ap, al in zip(act_prediction, act_label):
        al = [x for x in al if x != -1]
        ap = ap[:len(al)]
        flat_act_pred.extend(al)
        flat_act_true.extend(ap)
    ars = adjusted_rand_score(flat_act_true, flat_act_pred)
    amis = adjusted_mutual_info_score(flat_act_true, flat_act_pred)
    homo, comp, vm = homogeneity_completeness_v_measure(flat_act_true, flat_act_pred)
    act_result = (ars, amis, homo, comp, vm)
    

    logging.info(f'satisfaction: test_result={sat_result}, intent: test_result={act_result}')
    logging.info(f'satisfaction details: {sat_result_details}')

    # detailed analyses
    avg_p = sum(pointer_list)/len(pointer_list)
    logging.info(f'average point value: {avg_p}')

    with open(save_path + '/results.txt', 'w') as outfile:
        for i in range(len(pointer_list)):
            mask = len(data[f'{args.eval_set}']["input_text"][i])
            outfile.write(f'{data[f"{args.eval_set}"]["input_text"][i]}\t{sat_label[i]}\t{sat_prediction[i]}\t{act_label[i][:mask]}\t{act_prediction[i][:mask]}\t{att_a_list[i][:mask]}\t{att_c_list[i][:mask]}\t{pointer_list[i]}\n')



def representive_words(input_text, act_res, act_label, class_num, stop_path, save_path):
    stop_words = set()
    with open(stop_path, 'r') as infile:
        for line in infile:
            stop_words.add(line.strip('\n'))

    
    clusters = [[] for _ in range(class_num)]
    cluster_map = [{} for _ in range(class_num)]
    for text, act, label in zip(input_text, act_res, act_label):
        for utt, cluster, true in zip(text, act, label):
            utt = ' '.join(utt.split('|||'))
            score = max(cluster)
            idx = cluster.index(score)
            clusters[idx].append((utt, score, true))
            if true not in cluster_map[idx]:
                cluster_map[idx][true] = 0
            cluster_map[idx][true] += 1
    
    
    outfile = open(save_path + '/cluster_utt.txt', 'w')
    all_words = {}
    i = 0
    for c in clusters:
        #cmap = sorted(cluster_map[i].items(), key=lambda x:x[1], reverse=True)[0][0]
        for s in c:
            outfile.write(f'{s[0]}\t{i}\t{s[2]}\n')
            for word in s[0].split(' '):
                if word not in all_words:
                    all_words[word] = 0
                all_words[word] += s[1]
        i += 1
    outfile.close()

    i = 0
    outfile = open(save_path + '/cluster.txt', 'w')
    for c in clusters:
        cluster_word = {}
        for s in c:
            for word in s[0].split(' '):
                if word not in cluster_word:
                    cluster_word[word] = 0
                cluster_word[word] += s[1]
        sorted_word = sorted(cluster_word.items(), key=lambda x:x[1], reverse=True)
        word_freq = []
        for word in sorted_word[:500]:
            if word[0] not in stop_words:
                continue
            word_freq.append((word[0], word[1]/all_words[word[0]]))
        ranked_word = sorted(word_freq, key=lambda x:x[1], reverse=True)

        for word in ranked_word[:50]:
            #print(word[0], word[1], i)
            outfile.write(f'{word[0]}\t{word[1]}\t{i}\n')
        i += 1
        outfile.write('=======================================\n')
    outfile.close()

