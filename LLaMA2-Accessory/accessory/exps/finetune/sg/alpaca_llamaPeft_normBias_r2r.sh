#!/bin/bash


pretrained_path='/path/to/alpaca_llamaPeft_normBias'
pretrained_type=consolidated
tokenizer_path="path/to/tokenizer.model"
data_config='path/to/alpaca.yaml'


data_parallel=sdp
model_parallel=1

exp_name=/path/to/exprs

echo "exp name: $exp_name"
mkdir -p "$exp_name"

CUDA_VISIBLE_DEVICES=$1 torchrun --master_port=1112 --nproc_per_node=4 main_finetune.py \
--output_dir "$exp_name" --epochs 2 --warmup_epochs 1 \
--batch_size 2 --accum_iter 2 --num_workers 4 \
--max_words 400 \
--lr 0.001 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_peft --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
--teacher_forcing \
--precision "tf32" \
2>&1 | tee -a "$exp_name"/output.log

echo "exp name: $exp_name"