#!/bin/bash
python metrics/multi_features_entropy/multifeature_entropy.py \
  --input /path/to/images \
  --output results/entropy_exact.csv \
  --color-mode gray \
  --spatial-mode auto