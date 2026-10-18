# Multi-Feature Entropy for JPEG/JPG images

Implementation of the **four low-level entropy components** described by Luo et al., *A Multi-Feature Framework for Quantifying Information Content of Optical Remote Sensing Imagery*, Remote Sensing 14(16):4068 (2022). https://doi.org/10.3390/rs14164068

This is an independently implemented Python script **following the published equations**, not a verified copy of the authors' source code. Important implementation conventions are recorded below.

## Install

```bash
pip install numpy scipy pillow
```

Python 3.10+ recommended. No GPU, training or inference required.

## Basic usage

Run the **complete four-feature metric** over all JPG/JPEG images in a directory (no resizing):

```bash
python multifeature_entropy.py \
  --input /path/to/images \
  --output results/entropy_exact.csv \
  --color-mode gray \
  --spatial-mode auto
```

`auto` is **EXACT**, and uses pair enumeration for small pixel classes and FFT autocorrelation for large ones. The FFT version can be computationally expensive at 4K, especially with many populated gray levels. It does NOT silently switch to sampling.

For an initial **fast, approximate** survey (not publication-equivalent to exact):

```bash
python multifeature_entropy.py \
  --input /path/to/images \
  --recursive \
  --output results/entropy_sample.csv \
  --color-mode gray \
  --spatial-mode sample \
  --sample-pairs 200000 --seed 42
```

To measure both grayscale and RGB (the CSV has **two rows per image**):

```bash
python multifeature_entropy.py \
  --input /path/to/images \
  --output results/entropy_gray_rgb.csv \
  --color-mode both \
  --spatial-mode fft
```

To run self-tests:

```bash
python multifeature_entropy.py --self-test
```

See `python multifeature_entropy.py --help` for all settings.

## Modes

| Option | Values | Meaning |
|---|---|---|
| `--color-mode` | `gray` (default), `rgb`, `both` | Paper's grayscale experiment setting; or RGB bandwise Eq. (8) sum; or both |
| `--spatial-mode` | `auto` (default) | **Exact:** direct pairs for small levels; FFT autocorrelation otherwise |
| | `pairs` | **Exact:** exhaustive pairs with a safety cap; use only small images/classes |
| | `fft` | **Exact:** zero-padded 2D autocorrelation (typically high memory/runtime on 4K) |
| | `sample` | **Approximate:** uniform random unordered pairs *with replacement*, biased plug-in entropy |
| | `off` | Do not evaluate spatial term; `H_total` left blank (partial total separately reported) |
| `--spatial-denominator` | `all256` (default), `occupied` | Average over Q=256 classes vs only levels observed in the image |
| `--lbp-compare` | `ge` (default), `gt` | LBP neighbor >= center per published Eq. (2); or > center as in the prose |
| `--weights` | four real numbers (default `1 1 1 1`) | Coefficients for Eq. (7): gray, contrast, neighborhood, spatial |
| `--recursive` | flag | Recursively include `.jpg`, `.jpeg` case-insensitively |
| `--fft-workers` | positive integer | FFT threads; default `1` |
| `--local-chunk-rows` | positive integer | Local feature extraction strip size |
| `--fft-chunk-rows` | positive integer | Process FFT autocorrelation lag histogram in bounded row chunks |
| `--on-error` | `continue` (default), `raise` | Error policy |

The `pairs` mode refuses an excessive number of point pairs instead of silently allocating huge arrays. Control the threshold with `--max-exact-pairs` (default 1,000,000 **per intensity value**).

## Mathematics

On an 8-bit single channel:

1. **Grayscale entropy**: `H_gray = -sum_q p(q) log2(p(q))` (paper Eq. 1).
2. **Contrast entropy**: build 8-bit, 3x3 Local Binary Pattern codes, then Shannon entropy over the 256 code frequencies (Eq. 2–3).
3. **Neighborhood topology**: for every pixel, compute the Shannon entropy of the gray-level frequencies **among eight neighboring pixels**, then average over all pixels (Eq. 4).
4. **Spatial distribution**: for each intensity `q`, make all distinct, unordered pairs of pixels with intensity `q`. Compute Euclidean distances, Shannon entropy of distance frequencies, then average across gray levels (Eq. 5–6). Squared Euclidean distances `dx²+dy²` are used as the histogram keys: they preserve **exactly the same equality classes** as Euclidean distances and avoid floating-point binning or arbitrary quantization.

Total: `H_total = w_gray*H_gray + w_contrast*H_contrast + w_nei*H_nei + w_spatial*H_spatial` (Eq. 7), with default weights all 1. For RGB, each component and total are **summed across R, G and B**, according to Eq. (8), not averaged.

**Important:** This is an entropy-derived index, *not an established physical amount of semantic bits/pixel*. Do not divide total entropy by pixel count again.

## Details to disclose in a paper

- **8-bit color:** the input JPG/JPEG is EXIF-orientation corrected and decoded by Pillow. `gray` uses Pillow's RGB-to-`L` conversion; this particular grayscale conversion was not specified in detail by Luo et al. Report it explicitly. The script doesn't perform full ICC color-profile management.
- **LBP comparator discrepancy:** published formula's threshold is >=0, whereas prose says "greater than". Default `ge` follows the equation. Run `gt` as a sensitivity analysis if useful.
- **LBP histogram probability:** frequencies are divided by the number of valid codes (image pixels), to obtain a normalized distribution. The article's prose describing `N_c` is not completely clear.
- **Neighborhood edge handling:** the nearest edge row/column is replicated, as the paper states.
- **Spatial denominator ambiguity:** for 8-bit images, `all256` treats `N_q=256`, including entropy zero for unobserved classes. The paper calls `N_q` the number of gray levels, without fully resolving all possible vs occupied classes. `occupied` is supported for sensitivity analysis. **Do not pool scores from different denominators.**
- **Spatial equal-distance grouping:** use the integer squared distance (same exact equality partition as the distance itself). Avoid arbitrary distance bins, which would change the metric.
- **FFT formulation:** non-circular (zero-padded) autocorrelation counts equal-gray pixel-pairs at each 2D displacement. Each unordered pair is counted once using positive row offsets and positive column offsets for zero row offset. Radial grouping of those counts yields exactly the Eq. (5) distance frequency histogram. Uses double precision and checks the total number of pairs against n(n-1)/2.
- **Sampling caveat:** the Shannon plug-in entropy obtained from finite random pairs is generally **downward biased**, especially for 4K images with a large distance-category alphabet. Increasing `--sample-pairs` reduces but does not automatically eliminate this bias. Results from `sample` should NOT be compared as if equivalent to `auto`/`fft`/`pairs` without convergence studies against exact images.
- **Resolution dependence:** results depend on image dimensions and sampling. For a density-conditioned analysis, use consistent resolutions and preprocessing. This program never resizes or crops images automatically.
- **Provenance:** the `.meta.json` sidecar records all parameters, files processed, errors, total runtime and exact/approximate mode.

## Output

One `.csv` with one row per image and selected color mode, and a `.meta.json` sidecar:

- `image`, `width`, `height`, `megapixels`, `color_mode`, `spatial_mode`, `status`, `error`
- `H_gray`, `H_contrast`, `H_neighborhood`, `H_spatial`, **`H_total`**
- `H_partial_without_spatial`: diagnostic score (not full four-feature entropy)
- `H_gray_R`, `H_gray_G`, ... `H_total_B` (when `rgb` or `both`)
- Number of exact/FFT/pair/sample gray levels, sampled/possible pair counts, denominator, processing time

Files are saved after **each image**. Failed files are recorded as error rows; `--on-error raise` instead stops on failure. Scores across `gray` and `rgb` should be analyzed **separately**.

## Practical 4K guidance

- **Faithful metric reporting:** `--color-mode gray --spatial-mode auto --spatial-denominator all256 --lbp-compare ge --weights 1 1 1 1`. Consider sensitivity checks with `occupied` and `gt`.
- **Explore dataset quickly:** use `--spatial-mode sample` for diagnostics only. Keep its output separate and compare the ranking against a smaller exact subset before relying on sample-based results.
- **Memory:** FFT autocorrelation for just one populated class spanning a synthetic 3840x2160 image used approximately 1.2 GB peak process RSS in a local test (environment dependent). A single full-coverage class took ~6.5 seconds in that test; typical multilevel images require many separate FFTs and are much slower. This is not an end-to-end 4K per-image runtime claim.
- **Tested:** built-in small-image pair-vs-FFT equivalence tests and CLI processing of one **synthetic** 3840x2160 JPEG in `sample` mode. Not validated against an official reference implementation by the paper authors.
