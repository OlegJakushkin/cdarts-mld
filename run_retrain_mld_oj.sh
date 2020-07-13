NGPUS=2
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m torch.distributed.launch --nproc_per_node=$NGPUS retrain.py \
    --dataset mld --n_classes 2 --init_channels 16 --stem_multiplier 1 \
    --arc_checkpoint '/headless/tmp/w1t/cdarts-mld/outputs/epoch_01.json' \
    --batch_size 3 --workers 1 --log_frequency 10 \
    --world_size $NGPUS --weight_decay 5e-4 \
    --distributed --dist_url 'tcp://127.0.0.1:26443' \
    --lr 0.015 --warmup_epochs 0 --epochs 700 \
    --cutout_length 16 --aux_weight 0.4 --drop_path_prob 0.3 \
    --label_smooth 0.0 --mixup_alpha 0 --model_checkpoint "/headless/tmp/w1t/cdarts-mld/model_12.pt"
