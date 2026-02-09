#!/bin/bash

./moth/run_model.py train \
  --image-dir data/sampled/sampled_2026-01-21 \
  --train-json data/sampled/sampled_2026-01-21_train.json \
  --valid-json data/sampled/sampled_2026-01-21_valid.json \
  --batch-size 2 \
  --limit 2 \
  --epochs 1
