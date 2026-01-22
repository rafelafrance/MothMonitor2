#!/bin/bash

./moth/data_wrangler.py sample \
    --dir-glob 'data/MothitorPics2025/**/*' \
    --sample-dir data/sampled_2026-01-21 \
    --scale 0.5 \
    --long-edge 1333 \
    --short-edge 800 \
    --seed 382253 \
    --samples 500
