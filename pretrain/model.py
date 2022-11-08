from transformers import AdamW, BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from tqdm import tqdm
import copy
from transformer import TransformerEncoder


def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout

class BERTBackbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        bert_name = config.get('bert_name', 'bert-base-uncased')
        cache_dir = config.get('cache_dir')
        self.bert = BertModel.from_pretrained(bert_name, cache_dir=cache_dir)
        self.d_model = 768 * 2

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(0).detach()
        outputs = self.bert(input_ids, attention_mask)
        h = universal_sentence_embedding(outputs[0], attention_mask)
        cls = outputs[1]

        out = torch.cat([cls, h], dim=-1)
        return out

'''
class HiTrans_GRU(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.drop_out1 = nn.Dropout(args.dropout)
        self.drop_out2 = nn.Dropout(args.dropout)
        self.private = BERTBackbone(bert_name=args.bert_name, cache_dir=args.cache_dir)
        d_model = self.private.d_model

        self.gru = nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True)

        self.class_num = 2
        self.encoder = TransformerEncoder(d_model, d_model*2, 8, 2, 0.1)
        self.act_classifier = MLP(d_model, self.class_num, d_model//2)
        self.sat_classifier = nn.Linear(d_model, 2)
        #self.sat_classifier = MLP(d_model, 2, d_model//2)
        
        init_params(self.act_classifier)
        init_params(self.encoder)
        init_params(self.sat_classifier)

    def forward(self, input_ids, act_seq=None, sat=None, **kwargs):
        self.gru.flatten_parameters()

        batch_size, dialog_len, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = act_seq.ne(-1).detach()

        private_out = self.private(input_ids=input_ids, **kwargs)
        private_out = private_out.view(batch_size, dialog_len, -1) 
        H = self.encoder(private_out, attention_mask)
        hidden = universal_sentence_embedding(H, attention_mask)
        H = self.drop_out1(H)

        _, hidden = self.gru(H)
        hidden = hidden.squeeze(0)
        
        hidden = self.drop_out2(hidden)

        act_res = self.act_classifier(H)
        sat_res = self.sat_classifier(hidden)

        if self.training:
            act_loss = F.cross_entropy(act_res.view(-1, self.class_num), act_seq.view(-1), ignore_index=-1)
            sat_loss = F.cross_entropy(sat_res, sat)
            return act_res, sat_res, act_loss, sat_loss

        return act_res, sat_res

'''
class HiTrans_GRU(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.drop_out1 = nn.Dropout(args.dropout)
        self.drop_out2 = nn.Dropout(args.dropout)
        self.private = BERTBackbone(bert_name=args.bert_name, cache_dir=args.cache_dir)
        d_model = self.private.d_model

        #self.gru = nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True)

        self.class_num = 2
        self.encoder = TransformerEncoder(d_model, d_model*2, 8, 2, 0.1)
        #self.act_classifier = MLP(d_model, self.class_num, d_model//2)
        self.act_classifier = nn.Linear(d_model, self.class_num)
        self.sat_classifier = nn.Linear(d_model, 2)
        #self.sat_classifier = MLP(d_model, 2, d_model//2)
        
        init_params(self.act_classifier)
        init_params(self.encoder)
        init_params(self.sat_classifier)

    def forward(self, input_ids, act_seq=None, sat=None, **kwargs):
        #self.gru.flatten_parameters()

        batch_size, dialog_len, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = act_seq.ne(-1).detach()

        private_out = self.private(input_ids=input_ids, **kwargs)
        private_out = private_out.view(batch_size, dialog_len, -1) 
        H = self.encoder(private_out, attention_mask)
        hidden = universal_sentence_embedding(H, attention_mask)
        private_out = self.drop_out1(private_out)

        #_, hidden = self.gru(H)
        #hidden = hidden.squeeze(0)
        
        hidden = self.drop_out2(hidden)

        act_res = self.act_classifier(private_out)
        sat_res = self.sat_classifier(hidden)

        if self.training:
            act_loss = F.cross_entropy(act_res.view(-1, self.class_num), act_seq.view(-1), ignore_index=-1)
            sat_loss = F.cross_entropy(sat_res, sat)
            return act_res, sat_res, act_loss, sat_loss

        return act_res, sat_res