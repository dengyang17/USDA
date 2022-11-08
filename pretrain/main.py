from train import train
import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mwoz_pretrain')
parser.add_argument('--model', type=str, default='BERT')
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--gpu', default="0 1 2", type=str,
                    help="Use CUDA on the device.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--epoch_num", default=20, type=int,
                    help="Maximum number of training epochs.")
parser.add_argument("--max_seq_len", default=64, type=int,
                    help="Maximum input sequence length.")
parser.add_argument("--dropout", default=0.4, type=float,
                    help="Drop-out rate.")
parser.add_argument('--rewrite_data', action='store_true',
                    help='Rewrite data pickle.')
parser.add_argument('--cache_dir', default='/projdata1/info_fil/ydeng/bert',
                    type=str, help="Cache directory for storing PLMs.")
parser.add_argument('--bert_name', default='bert-base-uncased',
                    type=str, help="BERT model name.")
args = parser.parse_args()

print('train data', args.data)
print('train model', args.model)

# Setup CUDA, GPU & distributed training
device, device_id = utils.set_cuda(args)
args.device = device
args.device_id = device_id

# Set seed
utils.set_seed(args.seed)

train(args)