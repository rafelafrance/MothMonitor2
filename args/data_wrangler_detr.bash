#!/bin/bash

./moth/data_wrangler.py detr \
  --bbox-json data/sampled/sampled_2026-01-21.json \
  --base-path data/sampled/sampled_2026-01-21 \
  --train-fract 0.6 \
  --valid-fract 0.2 \
  --seed 382253
