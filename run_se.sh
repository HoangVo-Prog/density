#!/bin/bash
python metrics/subband_entropy/measure_subband_entropy.py \
  --input /path/to/images \
  --output ./subband_entropy_results \
  --mode both \
  --aggregation both \
  --tile-size 512 \
  --stride 512 \
  --matched-long-edge 2048 \
  --levels 3 --order 3 \
  --backend space --residuals none \
  --plots