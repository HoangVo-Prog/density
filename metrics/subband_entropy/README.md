# Subband Entropy Batch Calculator

**Script:** `measure_subband_entropy.py`

Recursively scans `.jpg` and `.jpeg` files (case-insensitive) in the specified directory and its subdirectories. No VQA annotations or models are required.

## Installation

```bash
python -m pip install -r requirements_subband_entropy.txt
```

Linux/WSL is recommended, as `pyrtools` may require compiled dependencies.

## Run All Modes

```bash
python measure_subband_entropy.py \
  --input /path/to/images \
  --output ./density_results \
  --mode both \
  --aggregation both \
  --tile-size 512 \
  --stride 512 \
  --matched-long-edge 2048 \
  --levels 3 --order 3 \
  --backend space --residuals none \
  --plots
```

The script generates four rows per image:

- **`native/global`**: Computes a single whole-image SE score at the original resolution.
- **`native/tiled`**: Processes the original image using 512 × 512 tiles and computes an area-weighted mean.
- **`matched/global`**: Downsamples the image if its longest edge exceeds 2048 pixels, then computes a whole-image SE score.
- **`matched/tiled`**: Computes an area-weighted mean over fixed-size tiles at the matched resolution.

**Important:** Whole-image (`global`) processing of 8K images can require substantial RAM. For safer execution, the recommended default is `--mode both --aggregation tiled`. Enable `--aggregation both` only after testing a small subset of images using `--limit 2`.

## Useful Command Examples

```bash
# Native/tiled only, without scanning subdirectories
python measure_subband_entropy.py -i ./images -o ./out \
  --mode native --aggregation tiled --no-recursive

# 50% tile overlap with overlap-aware weighting
python measure_subband_entropy.py -i ./images -o ./out \
  --mode native --aggregation tiled --tile-size 512 --stride 256

# Smoke test all four modes on two images
python measure_subband_entropy.py -i ./images -o ./smoke \
  --mode both --aggregation both --limit 2

# Whole-image processing only
python measure_subband_entropy.py -i ./images -o ./global \
  --mode native --aggregation global

# Alternative residual/backend settings for sensitivity analysis
python measure_subband_entropy.py -i ./images -o ./alt \
  --mode native --aggregation tiled --backend freq --residuals both
```

## Output Files

- **`image_density.csv`**: One row per `(image, resolution_mode, aggregation_mode)`, including `subband_entropy`, `entropy_L/a/b`, image dimensions, runtime, configuration parameters, and SHA-256 hashes.
- **`tile_density.csv`**: Tile coordinates, individual SE scores, and effective pixel-area weights.
- **`dataset_density_summary.csv`**: Mean, median, standard deviation, and quantile statistics grouped by the four processing modes.
- **`errors.csv`**: Image loading and measurement errors.
- **`run_config.json`**: Library versions, execution parameters, and methodological descriptions.
- **`distribution_*.png`**: Distribution histograms generated when `--plots` is enabled.

## Measurement Methodology and Interpretation

1. **Color-space conversion:** Convert RGB images to CIELab using `scikit-image`.

2. **Multiscale decomposition:** Apply a steerable pyramid decomposition to each CIELab channel using `pyrtools`, with three pyramid levels and four orientations by default.

3. **Subband entropy estimation:** For each subband containing \(N\) coefficients, construct a histogram with `ceil(sqrt(N))` bins spanning the minimum and maximum coefficient values. Compute the corresponding base-2 Shannon entropy.

4. **Channel aggregation:** Sum the entropy values across subbands within each channel, then compute a weighted combination using CIELab channel weights of `(L, a, b) = (0.84, 0.08, 0.08)`.

5. **Residual handling:** By default, `--residuals none` includes only orientation subbands. The options `both`, `lowpass`, and `highpass` are available for sensitivity analysis.

6. **Tiled aggregation:** Apply reflection padding to partial tiles and calculate an effective-area-weighted mean. When tiles overlap, each pixel's contribution is distributed equally among all tiles containing that pixel, preventing overlap regions from being overcounted.

### Interpretation

The resulting score should be interpreted as an **entropy-derived proxy for visual clutter**, rather than a direct measure of semantic information or information density in bits per pixel.

In particular:

- The score is **not equivalent to bits per pixel**.
- It does **not directly measure semantic information content**.
- It captures statistical variation in multiscale, orientation-sensitive image representations, rather than explicitly measuring object-level spatial distribution.
- Absolute scores obtained using `global` and `tiled` aggregation should **not be assumed to be directly comparable**.

### Scientific Caveat

This implementation follows the published methodology of Rosenholtz et al. (2007), but numerical equivalence with the authors' original MATLAB implementation has not yet been established.

Several implementation choices—including histogram bin-edge conventions, pyramid backend selection, and residual subband inclusion—must be explicitly specified and validated before using the measurements in a scientific publication.

Additional diagnostic experiments are recommended to assess sensitivity to image noise, repetitive textures, JPEG compression quality, resizing, and exposure variations. Human inspection should also be conducted before finalizing the measurement protocol.

**References:**

- Original paper: https://doi.org/10.1167/7.2.17
- Official MATLAB implementation: https://dspace.mit.edu/entities/publication/cbf48538-96a8-401c-a82a-2179ade7e6a8
- Steerable pyramid implementation (`pyrtools`): https://pyrtools.readthedocs.io/en/stable/

## Testing

```bash
python -m pip install pytest
python -m pytest -q test_subband_entropy.py
```

The bundled tests cover histogram entropy computation, tile coverage, image discovery, argument validation, and input/output behavior across all four processing modes.

The end-to-end tests use a **stubbed pyramid implementation** to verify the processing pipeline. Therefore, they do **not** establish numerical correctness or equivalence with either `pyrtools` or the original MATLAB implementation.

Numerical validation using diagnostic images should be performed separately after installing `pyrtools`.