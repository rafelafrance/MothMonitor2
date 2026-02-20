#!/bin/bash

./moth/data_wrangler.py sample \
  --dir-glob 'data/MothitorPics2025/**/*' \
  --sample-dir data/sampled_2026-02-18 \
  --scale 0.5 \
  --width 1333 \
  --height 800 \
  --seed 993669 \
  --samples 600
