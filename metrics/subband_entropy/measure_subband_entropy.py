#!/usr/bin/env python3
"""Question-independent Subband Entropy for JPEG image folders.

Based on Rosenholtz, Li & Nakano, *Measuring Visual Clutter*,
Journal of Vision 7(2):17 (2007), doi:10.1167/7.2.17.

Faithful to the paper at the level of RGB->CIELab, steerable-pyramid
subbands, per-subband histogram Shannon entropy with sqrt(N) bins, and
L/a/b weights (0.84,0.08,0.08). Pyramid implementation, exact histogram
bin-edge convention, residual treatment, and global/tiled aggregation
are explicitly parameterized, and have NOT been established as an
exact numerical reproduction of the original MATLAB implementation.

The output is a dimensionless subband-entropy-based score, NOT bits/pixel.
The tiled score is a research-protocol adaptation, NOT the paper's score.

Install: pip install numpy Pillow scikit-image pyrtools
Example:
  python measure_subband_entropy.py --input /data/images --output ./results \
    --mode both --aggregation both --tile-size 512
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageOps, ImageCms
from skimage.color import rgb2lab

# Import pyrtools lazily; '--help' works even if it is not installed.

PAPER_WEIGHTS = (0.84, 0.08, 0.08)
EXTENSIONS = {'.jpg', '.jpeg'}
APP_VERSION = '1.0.0'


@dataclass(frozen=True)
class MetricSettings:
    levels: int = 3
    order: int = 3
    backend: str = 'space'
    residuals: str = 'none'
    weights: tuple[float, float, float] = PAPER_WEIGHTS


def _check_pyrtools() -> None:
    try:
        import pyrtools  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing 'pyrtools'. Install dependencies with: "
            'python -m pip install numpy Pillow scikit-image pyrtools'
        ) from exc


def entropy_of_subband(coefficients: np.ndarray) -> float:
    """Shannon entropy of empirical coefficient bins, in bits/bin-symbol.

    The historical paper uses sqrt(number of coefficients) histogram bins.
    We choose CEIL for integer bins; MATLAB convention remains to be audited.
    This entropy is NOT divided by the number of pixels.
    """
    a = np.asarray(coefficients, dtype=np.float64)
    if a.size == 0:
        return 0.0
    if not np.isfinite(a).all():
        raise ValueError('Steerable pyramid produced non-finite coefficients')
    low, high = float(a.min()), float(a.max())
    # Ignore roundoff-scale variations (e.g., the numerical response of
    # a zero-mean filter on a constant image).
    if high - low <= 1e-10 * max(1.0, abs(low), abs(high)):
        return 0.0
    bins = max(2, math.ceil(math.sqrt(a.size)))
    counts, _ = np.histogram(a, bins=bins, range=(low, high))
    counts = counts[counts > 0].astype(np.float64)
    p = counts / float(a.size)
    return float(-np.dot(p, np.log2(p)))


def subband_entropy_rgb(rgb: np.ndarray, settings: MetricSettings) -> dict:
    """Compute weighted Subband Entropy of one uint8 RGB window.

    By default, includes all oriented subbands at a fixed 3 scales and
    4 orientations (order=3). Residual high/low pass can be enabled.
    A 64px minimum per side is an implementation guard, not paper preprocessing.
    """
    import pyrtools as pt

    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected a uint8 RGB image with shape (H,W,3)')
    h, w = rgb.shape[:2]
    if min(h, w) < 64:
        raise ValueError(
            f'Image/window too small ({w}x{h}); need at least 64px in each dimension'
        )
    lab = rgb2lab(rgb)
    channel_scores: list[float] = []
    total_subbands = 0
    for channel in range(3):
        data = np.ascontiguousarray(lab[:, :, channel], dtype=np.float64)
        if settings.backend == 'space':
            pyr = pt.pyramids.SteerablePyramidSpace(
                data, height=settings.levels, order=settings.order,
                edge_type='reflect1',
            )
        else:
            pyr = pt.pyramids.SteerablePyramidFreq(
                data, height=settings.levels, order=settings.order,
                is_complex=False,
            )
        keys = sorted(key for key in pyr.pyr_coeffs if isinstance(key, tuple))
        if settings.residuals in ('highpass', 'both'):
            keys.append('residual_highpass')
        if settings.residuals in ('lowpass', 'both'):
            keys.append('residual_lowpass')
        if not keys:
            raise RuntimeError('No pyramid subbands were produced')
        # A numerically constant Lab channel is always zero-information.
        # This also avoids reporting entropy of sub-picounit FFT/convolution noise.
        channel_scores.append(
            0.0 if float(data.max() - data.min()) <= 1e-10 else
            math.fsum(entropy_of_subband(pyr.pyr_coeffs[k]) for k in keys)
        )
        total_subbands += len(keys)
        del pyr, data
    score = math.fsum(w * h for w, h in zip(settings.weights, channel_scores))
    return {
        'subband_entropy': float(score),
        'entropy_L': channel_scores[0],
        'entropy_a': channel_scores[1],
        'entropy_b': channel_scores[2],
        'n_subbands_total': total_subbands,
    }


def find_jpegs(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob('*') if recursive else root.iterdir()
    return sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix().casefold(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path: Path, icc_policy: str) -> tuple[Image.Image, str]:
    """EXIF upright orientation; optionally convert embedded ICC to sRGB."""
    with Image.open(path) as raw:
        oriented = ImageOps.exif_transpose(raw)
        icc = oriented.info.get('icc_profile')
        if icc and icc_policy != 'ignore':
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_profile = ImageCms.createProfile('sRGB')
                rgb = ImageCms.profileToProfile(
                    oriented, src_profile, dst_profile,
                    outputMode='RGB', renderingIntent=0,
                )
                state = 'converted_to_srgb'
            except Exception as exc:
                if icc_policy == 'strict':
                    raise ValueError(f'ICC conversion failed: {exc}') from exc
                rgb = oriented.convert('RGB')
                state = 'icc_failed_fallback_rgb'
        else:
            rgb = oriented.convert('RGB')
            state = 'icc_ignored' if icc else 'no_icc'
    return rgb, state


def matched_resize(im: Image.Image, max_long_edge: int) -> Image.Image:
    w, h = im.size
    biggest = max(w, h)
    if biggest <= max_long_edge:
        return im
    scale = max_long_edge / biggest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return im.resize((new_w, new_h), Image.Resampling.LANCZOS)


def tile_starts(dimension: int, stride: int) -> range:
    return range(0, dimension, stride)


def inverse_coverage_1d(length: int, tile_size: int, stride: int) -> np.ndarray:
    """Pixel's fair share when overlapping tiles cover a coordinate."""
    diff = np.zeros(length + 1, dtype=np.int32)
    for start in tile_starts(length, stride):
        diff[start] += 1
        diff[min(length, start + tile_size)] -= 1
    n_cover = np.cumsum(diff[:-1])
    if not np.all(n_cover > 0):
        raise ValueError('Tiling leaves uncovered pixels; reduce --stride')
    return 1.0 / n_cover


def pad_tile_to_square(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h > size or w > size:
        raise ValueError('Cropping bug: image tile is larger than tile size')
    if h == size and w == size:
        return rgb
    # Explicitly a boundary adaptation: padded pixels are not independent data.
    return np.pad(rgb,
                  ((0, size - h), (0, size - w), (0, 0)),
                  mode='reflect')


def full_image_score(im: Image.Image, cfg: MetricSettings) -> dict:
    return subband_entropy_rgb(np.asarray(im, dtype=np.uint8), cfg)


def tiled_image_score(
    im: Image.Image,
    cfg: MetricSettings,
    tile_size: int,
    stride: int,
    tile_writer: Optional[csv.DictWriter],
    tile_identity: dict,
) -> dict:
    w, h = im.size
    inv_x = inverse_coverage_1d(w, tile_size, stride)
    inv_y = inverse_coverage_1d(h, tile_size, stride)
    # Fast cumulative summation of fair exposure weights per axis.
    psx = np.concatenate(([0.0], np.cumsum(inv_x)))
    psy = np.concatenate(([0.0], np.cumsum(inv_y)))
    weighted_sum = np.zeros(4, dtype=np.float64)
    weighted_squared = 0.0
    total_weight = 0.0
    values: list[float] = []
    n_tiles = 0
    for y in tile_starts(h, stride):
        for x in tile_starts(w, stride):
            right = min(w, x + tile_size)
            bottom = min(h, y + tile_size)
            crop = im.crop((x, y, right, bottom))
            rgb = pad_tile_to_square(np.asarray(crop, dtype=np.uint8), tile_size)
            scores = subband_entropy_rgb(rgb, cfg)
            # Each original pixel contributes exactly 1 unit of mass across
            # ALL tiles covering it. For stride==tile_size, this is raw area.
            weight = float((psx[right] - psx[x]) * (psy[bottom] - psy[y]))
            vector = np.array([
                scores['subband_entropy'], scores['entropy_L'],
                scores['entropy_a'], scores['entropy_b'],
            ], dtype=np.float64)
            weighted_sum += weight * vector
            weighted_squared += weight * float(vector[0] ** 2)
            total_weight += weight
            values.append(float(scores['subband_entropy']))
            n_tiles += 1
            if tile_writer is not None:
                tile_writer.writerow({
                    **tile_identity, 'tile_index': n_tiles - 1,
                    'x': x, 'y': y, 'tile_width': right - x,
                    'tile_height': bottom - y,
                    'weight_effective_area': weight,
                    'was_reflect_padded': int((right-x < tile_size) or (bottom-y < tile_size)),
                    **scores,
                })
    if not math.isclose(total_weight, w * h, rel_tol=1e-9, abs_tol=1e-5):
        raise RuntimeError(
            f'Coverage weights do not sum to image area: {total_weight} vs {w*h}'
        )
    mean = weighted_sum / total_weight
    variance = max(0.0, weighted_squared / total_weight - mean[0] ** 2)
    return {
        'subband_entropy': float(mean[0]),
        'entropy_L': float(mean[1]),
        'entropy_a': float(mean[2]),
        'entropy_b': float(mean[3]),
        'n_subbands_total': n_tiles * (
            3 * (cfg.levels * (cfg.order + 1) +
                 (1 if cfg.residuals in ('lowpass', 'highpass') else
                  2 if cfg.residuals == 'both' else 0))
        ),
        'n_tiles': n_tiles,
        'tile_min': float(np.min(values)),
        'tile_median': float(np.median(values)),
        'tile_max': float(np.max(values)),
        'tile_std_area_weighted': float(math.sqrt(variance)),
    }


IMAGE_COLUMNS = [
    'image_id', 'relative_path', 'sha256_file', 'file_bytes',
    'original_width', 'original_height', 'original_megapixels',
    'measured_width', 'measured_height', 'icc_status',
    'resolution_mode', 'aggregation_mode',
    'subband_entropy', 'entropy_L', 'entropy_a', 'entropy_b',
    'n_subbands_total', 'n_tiles', 'tile_min', 'tile_median',
    'tile_max', 'tile_std_area_weighted', 'seconds',
    'levels', 'orientations', 'backend', 'residuals',
    'tile_size', 'stride', 'matched_long_edge',
]
TILE_COLUMNS = [
    'image_id', 'relative_path', 'resolution_mode', 'aggregation_mode',
    'tile_index', 'x', 'y', 'tile_width', 'tile_height',
    'weight_effective_area', 'was_reflect_padded',
    'subband_entropy', 'entropy_L', 'entropy_a', 'entropy_b',
    'n_subbands_total',
]
ERROR_COLUMNS = ['relative_path', 'error_type', 'message', 'traceback']
SUMMARY_COLUMNS = [
    'resolution_mode', 'aggregation_mode', 'n_images', 'mean', 'median',
    'std', 'min', 'max', 'q10', 'q25', 'q75', 'q90',
    'mean_runtime_seconds',
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Subband Entropy of all JPG/JPEG in a folder (native/matched, global/tiled).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', '-i', type=Path, required=True, help='Input image folder')
    p.add_argument('--output', '-o', type=Path, required=True, help='Output folder')
    p.add_argument('--recursive', action=argparse.BooleanOptionalAction,
                   default=True, help='Include subfolders')
    p.add_argument('--mode', choices=['native', 'matched', 'both'],
                   default='both', help='Resolution protocol')
    p.add_argument('--aggregation', choices=['global', 'tiled', 'both'],
                   default='tiled', help='Spatial aggregation protocol; tiled is safer for 8K')
    p.add_argument('--matched-long-edge', type=int, default=2048,
                   help='Longest side limit, downsampling only; same field of view')
    p.add_argument('--tile-size', type=int, default=512,
                   help='Fixed square tile extent in pixels')
    p.add_argument('--stride', type=int, default=None,
                   help='Tile stride. Default = tile size; <=tile size to prevent gaps.')
    p.add_argument('--levels', type=int, default=3,
                   help='Number of steerable pyramid scales')
    p.add_argument('--order', type=int, choices=[1, 3, 5], default=3,
                   help='Steerable order: number of orientations = order + 1')
    p.add_argument('--backend', choices=['space', 'freq'], default='space',
                   help='Pyramid method; space uses reflective edges, freq uses circular')
    p.add_argument('--residuals', choices=['none', 'lowpass', 'highpass', 'both'],
                   default='none', help='Additional high/low pass residual subbands')
    p.add_argument('--icc-policy', choices=['convert', 'ignore', 'strict'],
                   default='convert', help='Embedded ICC handling')
    p.add_argument('--save-tiles', action=argparse.BooleanOptionalAction,
                   default=True, help='Save per-tile scores to tile_density.csv')
    p.add_argument('--hash-files', action=argparse.BooleanOptionalAction,
                   default=True, help='SHA256 files for reproducibility')
    p.add_argument('--plots', action='store_true',
                   help='Create histogram plot per combination (requires matplotlib)')
    p.add_argument('--limit', type=int, default=0,
                   help='Only process first N matching images, for smoke tests')
    p.add_argument('--fail-fast', action='store_true',
                   help='Abort rather than logging and continuing on invalid images')
    args = p.parse_args(argv)
    if not args.input.is_dir():
        p.error(f'Not an image folder: {args.input}')
    if args.matched_long_edge < 64:
        p.error('--matched-long-edge must be >=64')
    if args.tile_size < 64:
        p.error('--tile-size must be >=64')
    if args.stride is None:
        args.stride = args.tile_size
    if not 1 <= args.stride <= args.tile_size:
        p.error('--stride must be in [1, --tile-size]')
    if args.levels < 1:
        p.error('--levels must be >=1')
    if args.limit < 0:
        p.error('--limit must be >=0')
    return args


def write_summary(rows: list[dict], destination: Path) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row['resolution_mode'], row['aggregation_mode'])].append(row)
    with destination.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (r_mode, a_mode), items in sorted(groups.items()):
            values = np.asarray([float(x['subband_entropy']) for x in items])
            runtime = np.asarray([float(x['seconds']) for x in items])
            writer.writerow({
                'resolution_mode': r_mode, 'aggregation_mode': a_mode,
                'n_images': len(items), 'mean': float(values.mean()),
                'median': float(np.median(values)),
                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'min': float(values.min()), 'max': float(values.max()),
                'q10': float(np.quantile(values, .10)),
                'q25': float(np.quantile(values, .25)),
                'q75': float(np.quantile(values, .75)),
                'q90': float(np.quantile(values, .90)),
                'mean_runtime_seconds': float(runtime.mean()),
            })


def maybe_plot(rows: list[dict], folder: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row['resolution_mode'], row['aggregation_mode'])].append(
            float(row['subband_entropy'])
        )
    for (resolution, aggregation), values in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=min(30, max(5, int(math.sqrt(len(values))))))
        ax.set_title(f'Subband Entropy — {resolution} / {aggregation} (n={len(values)})')
        ax.set_xlabel('Subband Entropy score (not bits/pixel)')
        ax.set_ylabel('Number of images')
        fig.tight_layout()
        fig.savefig(folder / f'distribution_{resolution}_{aggregation}.png', dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> int:
    _check_pyrtools()
    cfg = MetricSettings(levels=args.levels, order=args.order,
                         backend=args.backend, residuals=args.residuals)
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    paths = find_jpegs(args.input, args.recursive)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print('No *.jpg or *.jpeg images found.', file=sys.stderr)
        return 2
    import PIL, skimage, pyrtools
    metadata = {
        'script_version': APP_VERSION,
        'metric_definition': 'Rosenholtz-inspired subband entropy; NOT MATLAB-equivalent validated',
        'score_unit': 'dimensionless entropy-derived score (not bits/pixel)',
        'tiled_score': 'effective-pixel-area-weighted average of fixed-size window scores',
        'source': 'Rosenholtz et al. 2007, doi:10.1167/7.2.17',
        'weights_L_a_b': list(cfg.weights),
        'histogram_bins': 'ceil(sqrt(number_of_coefficients))',
        'histogram_range': 'min-to-max per subband',
        'pyramid_coefficients': 'all (scale,orientation) bands, optional residuals',
        'arguments': {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'Pillow': PIL.__version__, 'scikit_image': skimage.__version__,
                     'pyrtools': getattr(pyrtools, '__version__', 'unknown')},
    }
    (folder / 'run_config.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    image_rows = []
    errors = []
    mode_list = ['native', 'matched'] if args.mode == 'both' else [args.mode]
    agg_list = ['global', 'tiled'] if args.aggregation == 'both' else [args.aggregation]

    with (folder / 'image_density.csv').open('w', newline='', encoding='utf-8') as image_out:
        image_writer = csv.DictWriter(image_out, fieldnames=IMAGE_COLUMNS)
        image_writer.writeheader()
        tile_out = None
        tile_writer = None
        if args.save_tiles and 'tiled' in agg_list:
            tile_out = (folder / 'tile_density.csv').open('w', newline='', encoding='utf-8')
            tile_writer = csv.DictWriter(tile_out, fieldnames=TILE_COLUMNS)
            tile_writer.writeheader()
        try:
            print(f'Found {len(paths)} JPEG images; {len(mode_list)*len(agg_list)} mode(s) each.')
            for i, path in enumerate(paths, 1):
                relative = path.relative_to(args.input).as_posix()
                try:
                    im, icc_status = read_rgb(path, args.icc_policy)
                    ow, oh = im.size
                    image_id = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]
                    sha = sha256_file(path) if args.hash_files else ''
                    for resolution_mode in mode_list:
                        view = (im if resolution_mode == 'native' else
                                matched_resize(im, args.matched_long_edge))
                        measured_w, measured_h = view.size
                        for aggregation_mode in agg_list:
                            t0 = time.perf_counter()
                            tile_id = {
                                'image_id': image_id, 'relative_path': relative,
                                'resolution_mode': resolution_mode,
                                'aggregation_mode': aggregation_mode,
                            }
                            if aggregation_mode == 'global':
                                scores = full_image_score(view, cfg)
                                scores.update({
                                    'n_tiles': 1, 'tile_min': '', 'tile_median': '',
                                    'tile_max': '', 'tile_std_area_weighted': '',
                                })
                            else:
                                scores = tiled_image_score(
                                    view, cfg, args.tile_size, args.stride,
                                    tile_writer, tile_id,
                                )
                            row = {
                                'image_id': image_id, 'relative_path': relative,
                                'sha256_file': sha, 'file_bytes': path.stat().st_size,
                                'original_width': ow, 'original_height': oh,
                                'original_megapixels': (ow * oh) / 1e6,
                                'measured_width': measured_w, 'measured_height': measured_h,
                                'icc_status': icc_status,
                                'resolution_mode': resolution_mode,
                                'aggregation_mode': aggregation_mode,
                                'seconds': time.perf_counter() - t0,
                                'levels': cfg.levels, 'orientations': cfg.order + 1,
                                'backend': cfg.backend, 'residuals': cfg.residuals,
                                'tile_size': args.tile_size if aggregation_mode == 'tiled' else '',
                                'stride': args.stride if aggregation_mode == 'tiled' else '',
                                'matched_long_edge': (args.matched_long_edge
                                                      if resolution_mode == 'matched' else ''),
                                **scores,
                            }
                            image_writer.writerow(row)
                            image_rows.append(row)
                            print(f'[{i}/{len(paths)}] {relative} | '
                                  f'{resolution_mode}/{aggregation_mode}: '
                                  f'{row["subband_entropy"]:.6f} '
                                  f'({row["seconds"]:.2f}s)', flush=True)
                except Exception as exc:
                    errors.append({
                        'relative_path': relative, 'error_type': type(exc).__name__,
                        'message': str(exc), 'traceback': traceback.format_exc(),
                    })
                    print(f'[ERROR {i}/{len(paths)}] {relative}: {exc}',
                          file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise
        finally:
            if tile_out is not None:
                tile_out.close()

    with (folder / 'errors.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        writer.writerows(errors)
    write_summary(image_rows, folder / 'dataset_density_summary.csv')
    if args.plots and image_rows:
        try:
            maybe_plot(image_rows, folder)
        except ImportError:
            print('Matplotlib not installed; skipping plots.', file=sys.stderr)
    print(f'DONE: {len(paths)} files, {len(image_rows)} successful rows, '
          f'{len(errors)} file errors. Output: {folder.resolve()}')
    return 0 if not errors else 1


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
