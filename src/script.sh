#!/bin/bash

lrs=("1e-5" "1e-6" "1e-7")
weight_decays=("0" "1e-5")
warmup_ratios=("0.1" "0.25" "0.33")
augmentation_thresholds=("0.0" "0.5")

for lr in "${lrs[@]}"; do
  for weight_decay in "${weight_decays[@]}"; do
    for warmup_ratio in "${warmup_ratios[@]}"; do
      for augmentation_threshold in "${augmentation_thresholds[@]}"; do
        command="python src\train.py --lr $lr --weight_decay $weight_decay --warmup_ratio $warmup_ratio --augmentation_threshold $augmentation_threshold"
        echo "$command"
      done
    done
  done
done