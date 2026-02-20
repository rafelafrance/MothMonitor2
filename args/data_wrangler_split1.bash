#!/bin/bash

./moth/data_wrangler.py split \
  --bbox-json data/sampled/sampled_2026-01-21.json \
  --base-path data/sampled/sampled_2026-01-21 \
  --train-fract 0.6 \
  --valid-fract 0.2 \
  --seed 113944
