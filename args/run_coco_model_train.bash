#!/bin/bash

./moth/run_coco_model.py train \
  --image-dir data/sampled/sampled_2026-01-21 \
  --train-json data/sampled/sampled_2026-01-21_train.json \
  --valid-json data/sampled/sampled_2026-01-21_valid.json \
  --best-checkpoint data/models/fastercnn_resnet50_2026-02-11a.pth \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --amsgrad \
  --epochs 50
