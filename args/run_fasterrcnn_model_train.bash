#!/bin/bash

./moth/run_fasterrcnn_model.py train \
  --image-dir data/sampled/sampled_2026-0121_0218 \
  --train-json data/sampled/sampled_2026-0121_0218_train.json \
  --valid-json data/sampled/sampled_2026-0121_0218_valid.json \
  --save-checkpoint data/models/fasterrcnn_resnet50_2026-02-20.pth \
  --optimizer-class adamw \
  --batch-size 4 \
  --learning-rate 3e-5 \
  --weight-decay 3e-3 \
  --epochs 100
