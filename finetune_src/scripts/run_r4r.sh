ob_type=cand
feedback=sample

features=vit-16-ori
ft_dim=512

ngpus=1
seed=0

outdir=../datasets/R2R/exprs/finetune/agent/r4r

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r4r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 30
      --max_instr_len 250
      --max_seq_len 800

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 32
      --optim adamW

      --ml_weight 0.1

      --feat_dropout 0.4
      --dropout 0.5"

# inference
CUDA_VISIBLE_DEVICES='0' torchrun --master_port 12229 --nproc_per_node 1 r2r/main.py $flag  \
      --llm_predict --stop_first --pretrained_path /path/to/pretrianed_path \
      --llama_type llama_peft --no_visual --pretrained_type consolidated \
      --tokenizer_path /path/to/tokenizer  --dtype fp16 \
      --test --submit \
