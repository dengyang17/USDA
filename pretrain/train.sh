#CUDA_VISIBLE_DEVICES=0,1 python main.py --model=BERT --gpu='0 1' --batch_size=24
#CUDA_VISIBLE_DEVICES=0,1 python main.py --model=BERT --gpu='0 1' --batch_size=24 --data=sgd_pretrain
#CUDA_VISIBLE_DEVICES=0,1 python main.py --model=BERT --gpu='0 1' --batch_size=24 --data=redial_pretrain
CUDA_VISIBLE_DEVICES=0,1 python main.py --model=BERT --gpu='0 1' --batch_size=24 --data=jddc_pretrain --bert_name=bert-base-chinese