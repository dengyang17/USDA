#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100000M
#SBATCH --partition=infofil01
#hostname
nvidia-smi


#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=mwoz --model=USDA_P
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=sgd --model=USDA_P
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=redial --model=USDA_P
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=jddc --model=USDA_P --bert_name=bert-base-chinese
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=16 --class_num=20 --data=sgd
#CUDA_VISIBLE_DEVICES=1,2 python main.py --gpu='0 1' --batch_size=16 --class_num=20 --data=jddc
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=16 --class_num=20 --data=redial
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=mwoz --model=USDA_P --pretrain_model=../pretrain/outputs/mwoz_pretrain_BERT/best_pretrain.pt
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=sgd --model=USDA_P --pretrain_model=../pretrain/outputs/sgd_pretrain_BERT/best_pretrain.pt
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --class_num=20 --data=redial --model=USDA_P --pretrain_model=../pretrain/outputs/redial_pretrain_BERT/best_pretrain.pt
CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --model=USDA_P --data=jddc --bert_name=bert-base-chinese --pretrain_model=../pretrain/outputs/jddc_pretrain_BERT/best_pretrain.pt --class_num=20
