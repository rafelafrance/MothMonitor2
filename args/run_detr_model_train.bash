#!/bin/bash

./moth/run_detr_model.py train \
  --train-json data/sampled/sampled_2026-01-21_train.json \
  --valid-json data/sampled/sampled_2026-01-21_valid.json \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --epochs 50
