# Subband Entropy: measurement-protocol audit

This suite tests whether image-level Subband Entropy (SE) measurements are stable
under controlled implementation changes. **It does not prove that SE measures
semantic image information, VQA difficulty, or that pyrtools reproduces the
original Rosenholtz MATLAB implementation.**

## Fast correctness tests

From the repo root:

```bash
python -m pip install -r requirements_metrics.txt pytest
python -m pytest -q metrics/subband_entropy/test_subband_entropy.py \
  metrics/subband_entropy/test_se_protocol.py
```

The tests verify exact pixel coverage and overlap mass, no missing windows,
full-window anchored tiling without padding, the isolation of padding rules
(`reflect` vs `edge` **on exactly the same crop**), exact agreement with the
production reflect-padded aggregation using a deterministic scoring stub,
resize-as-downsampling semantics, rank-statistic behavior, and complete CSV/JSON
creation. A real-pyramid constant-image/CIELab-weight test runs when `pyrtools`
is installed. Stub-based results are **not** SE numerical validation.

## Paired audit on real JPEGs

```bash
python metrics/subband_entropy/audit_se_protocol.py \
  --input /path/to/images --output ./se_protocol_audit \
  --mode both --limit 30 --tile-size 512 --stride 512 \
  --matched-long-edge 2048 --levels 3 --order 3 \
  --backend space --residuals none --grid-phase
```

Outputs:
- `protocol_per_image.csv`: score from all protocols, padding exposure and paired deltas.
- `protocol_summary.json`: mean/absolute differences, Spearman rho, mean and
  maximum rank shift per resolution mode, plus resize comparisons on identical images.
- `errors.csv`: explicit records when an image is smaller than a full tile or fails.

This audit is CPU-expensive: it recomputes the SE pyramid for each alternative.
Start with `--limit 5` and increase to `--limit 30` when the first run completes.
Do not mix reports produced with different pyramid settings or tile sizes.

## What each contrast rules out

| Contrast | What is held fixed? | What it isolates |
|---|---|---|
| `same_crop_edge` vs `legacy_reflect` | Original crop pixels, crop positions and effective-area weights | **Padding-rule sensitivity**; if rank shifts, the chosen boundary extrapolation matters |
| `full_tile_anchor` vs `legacy_reflect` | Original image, tile size, SE implementation | Combined window-position / coverage / padding effect (**not** a causal estimate of padding alone) |
| `full_tile_even` vs `full_tile_anchor` | Full-size, unpadded windows, coverage-normalized aggregation | Grid-position / phase sensitivity |
| `native` vs `matched` | Same source and field of view | Sensitivity to changing pixel scale |

**Do not treat ρ close to 1 as proof of semantic validity.** If the same-crop
padding contrast is small but the anchored contrast is large, grid placement or
selection of original content is a more plausible explanation than the padding
rule itself. If both are large, further isolate boundary effects with visual
heatmaps and same-coordinate comparisons.

## Recommended experimental decisions

- Pre-register sensitivity thresholds *before* looking at held-out benchmark results
  (e.g. ρ ≥ 0.95 and maximum rank shift ≤ 5 on a pilot); these are choices for
  screening **not statistical or theoretical correctness thresholds**.
- Report all three paired comparisons, numbers of images actually measured,
  absolute changes and ranking changes. Flag a protocol as **sensitive** rather
  than automatically labeling the metric "incorrect".
- Use separate natural-image studies for additive noise, JPEG quality changes,
  texture repetition, and matched scale, then verify interpretation by inspecting
  image/region exemplars. No general monotonicity of entropy is guaranteed for
  these perturbations.
- Do not present a native/global minus native/tiled score as a bug: histogram
  sizes and subband coefficient populations differ between protocols.
- Original MATLAB numerical parity remains a **separate outstanding task**.
