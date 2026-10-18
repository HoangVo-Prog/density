#!/bin/bash
python metrics/multifeatures_entropy/multifeature_entropy.py \
  --input /path/to/images \
  --output multifeature_entropy_results/entropy_exact.csv \
  --color-mode gray \
  --spatial-mode auto