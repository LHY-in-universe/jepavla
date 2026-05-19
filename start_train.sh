#!/bin/bash
source /root/miniconda3/bin/activate Evo1
cd /root/autodl-tmp/jepaevotest/jepavla

python -u train_vla.py   --data-root /root/autodl-tmp/jepaevotest/datasets/Evo1_MetaWorld_Dataset   --config configs/train_vla_metaworld.yaml   --stage A   --dataset-backend lerobot   --task-level hard   --device cuda   --num-workers 1   --max-steps 250000   --log-every 100   --eval-every 5000   --eval-batches 50   --save-every 5000   > train_stage_a.log 2>&1
