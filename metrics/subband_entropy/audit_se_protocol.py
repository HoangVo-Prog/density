#!/usr/bin/env python3
"""Paired, question-independent sensitivity audit for Subband Entropy.

Does NOT change the original SE estimator or assert scientific validity.
Compare identical decoded images under controlled boundary protocols:
  reflect: production legacy partial windows + reflect padding;
  edge:    IDENTICAL partial windows + replication padding (padding-rule control);
  anchor:  full-size windows with the last window anchored at the image border;
  even:    full-size windows spaced evenly, a grid-phase control.

The anchor vs reflect contrast combines boundary-content selection and padding;
only edge vs reflect holds crop coordinates and original pixels fixed.

Usage (from repository root):
  python metrics/subband_entropy/audit_se_protocol.py \\
    --input /path/to/jpegs --output ./se_protocol_audit \\
    --mode native --limit 30 --tile-size 512 --stride 512 --grid-phase
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr

import measure_subband_entropy as se


STRATEGIES = ('reflect', 'edge', 'anchor', 'even')
COLUMNS = (
    'relative_path', 'image_sha256', 'resolution_mode', 'width', 'height',
    'legacy_reflect', 'same_crop_edge', 'full_tile_anchor', 'full_tile_even',
    'edge_minus_reflect', 'anchor_minus_reflect', 'even_minus_anchor',
    'padded_tiles', 'n_tiles_reflect', 'padded_effective_area_fraction',
    'artificial_pixel_fraction', 'n_tiles_anchor',
)


def starts(length: int, size: int, stride: int, strategy: str) -> list[int]:
    if not (length > 0 and size > 0 and 1 <= stride <= size):
        raise ValueError('Require length/size > 0 and 1 <= stride <= size')
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown tiling strategy: {strategy}')
    if strategy in ('reflect', 'edge'):
        return list(range(0, length, stride))
    if length < size:
        raise ValueError('Full-size anchored tiling requires each image dimension >= tile size')
    last = length - size
    if last == 0:
        return [0]
    if strategy == 'anchor':
        locations = list(range(0, last + 1, stride))
        if locations[-1] != last:
            locations.append(last)
        return locations
    n = math.ceil(last / stride) + 1
    return sorted(set(round(i * last / (n - 1)) for i in range(n)))


def fair_axis_weights(length: int, size: int, starts_: list[int]) -> np.ndarray:
    """Each pixel gives a total of one unit over all covering tiles."""
    diff = np.zeros(length + 1, dtype=np.int64)
    for start in starts_:
        if start < 0 or start >= length:
            raise ValueError('Window starts outside image')
        diff[start] += 1
        diff[min(length, start + size)] -= 1
    counts = np.cumsum(diff[:-1])
    if np.any(counts < 1):
        raise ValueError('The proposed windows do not cover the entire image')
    return 1.0 / counts


def pad_partial(tile: np.ndarray, size: int, strategy: str) -> np.ndarray:
    h, w = tile.shape[:2]
    if (h, w) == (size, size):
        return tile
    if strategy not in ('reflect', 'edge'):
        raise AssertionError('Anchored strategies must not pad')
    return np.pad(tile, ((0, size-h), (0, size-w), (0, 0)), mode=strategy)


def measure_strategy(
    im: Image.Image, size: int, stride: int, strategy: str,
    score_fn: Callable[[np.ndarray], float],
) -> dict[str, float | int]:
    w, h = im.size
    xs, ys = starts(w, size, stride, strategy), starts(h, size, stride, strategy)
    inv_x = fair_axis_weights(w, size, xs)
    inv_y = fair_axis_weights(h, size, ys)
    psx = np.r_[0.0, np.cumsum(inv_x)]
    psy = np.r_[0.0, np.cumsum(inv_y)]
    score_total = 0.0
    area_total = 0.0
    padded_area = 0.0
    artificial_pixels = 0
    n_padded = 0
    for y in ys:
        for x in xs:
            right, bottom = min(w, x + size), min(h, y + size)
            rgb = np.asarray(im.crop((x, y, right, bottom)).convert('RGB'), dtype=np.uint8)
            if right-x < size or bottom-y < size:
                n_padded += 1
                artificial_pixels += size*size - (right-x)*(bottom-y)
            rgb = pad_partial(rgb, size, strategy)
            score = float(score_fn(rgb))
            if not math.isfinite(score):
                raise ValueError('Nonfinite SE score')
            area = float((psx[right]-psx[x]) * (psy[bottom]-psy[y]))
            score_total += area * score
            area_total += area
            if right-x < size or bottom-y < size:
                padded_area += area
    if not math.isclose(area_total, w*h, abs_tol=1e-5, rel_tol=1e-10):
        raise AssertionError(f'Incorrect area sum: {area_total}, expected {w*h}')
    n_tiles = len(xs)*len(ys)
    return dict(
        score=score_total/area_total, n_tiles=n_tiles, padded_tiles=n_padded,
        padded_effective_area_fraction=padded_area/area_total,
        artificial_pixel_fraction=artificial_pixels/(size*size*n_tiles),
    )


def ranking_summary(rows: list[dict], alt: str, base: str = 'legacy_reflect') -> dict:
    pairs = [(float(r[base]), float(r[alt])) for r in rows
             if r.get(base) not in ('', None) and r.get(alt) not in ('', None)]
    n = len(pairs)
    summary = {'n': n, 'base': base, 'alternative': alt}
    if not n:
        return summary
    a, b = np.asarray(pairs, dtype=np.float64).T
    d = b-a
    summary.update(mean_delta=float(d.mean()), mean_abs_delta=float(np.abs(d).mean()),
                   max_abs_delta=float(np.abs(d).max()))
    if n >= 3 and np.ptp(a) > 1e-12 and np.ptp(b) > 1e-12:
        rho = float(spearmanr(a, b).statistic)
        shifts = np.abs(rankdata(-a, method='average')-rankdata(-b, method='average'))
        summary.update(spearman_rho=rho, max_rank_shift=float(shifts.max()),
                       mean_rank_shift=float(shifts.mean()))
    else:
        summary.update(spearman_rho=None, max_rank_shift=None, mean_rank_shift=None)
    return summary


def audit(args: argparse.Namespace) -> dict:
    se._check_pyrtools()
    images = se.find_jpegs(args.input, recursive=True)
    if args.limit:
        images = images[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    modes = ('native', 'matched') if args.mode == 'both' else (args.mode,)
    cfg = se.MetricSettings(levels=args.levels, order=args.order,
                            backend=args.backend, residuals=args.residuals)
    for ix, path in enumerate(images, 1):
        relative = path.relative_to(args.input).as_posix()
        try:
            native, _ = se.read_rgb(path, 'convert')
            for mode in modes:
                im = native if mode == 'native' else se.matched_resize(native, args.matched_long_edge)
                w, h = im.size
                if min(w, h) < args.tile_size:
                    errors.append(dict(image=relative, mode=mode,
                                       reason='dimension below tile size; full-size strategy undefined'))
                    continue
                score_fn = lambda rgb: se.subband_entropy_rgb(rgb, cfg)['subband_entropy']
                # The real production baseline, not a rewritten approximation.
                original = se.tiled_image_score(im, cfg, args.tile_size, args.stride, None, {})
                edge = measure_strategy(im, args.tile_size, args.stride, 'edge', score_fn)
                anchor = measure_strategy(im, args.tile_size, args.stride, 'anchor', score_fn)
                even = (measure_strategy(im, args.tile_size, args.stride, 'even', score_fn)
                        if args.grid_phase else None)
                # Geometry is identical between reflect and edge: use edge for
                # padding accounting, never attribute anchor changes to padding alone.
                current = float(original['subband_entropy'])
                rows.append(dict(
                    relative_path=relative, image_sha256=se.sha256_file(path),
                    resolution_mode=mode, width=w, height=h,
                    legacy_reflect=current, same_crop_edge=edge['score'],
                    full_tile_anchor=anchor['score'],
                    full_tile_even=even['score'] if even else '',
                    edge_minus_reflect=edge['score']-current,
                    anchor_minus_reflect=anchor['score']-current,
                    even_minus_anchor=even['score']-anchor['score'] if even else '',
                    padded_tiles=edge['padded_tiles'], n_tiles_reflect=edge['n_tiles'],
                    padded_effective_area_fraction=edge['padded_effective_area_fraction'],
                    artificial_pixel_fraction=edge['artificial_pixel_fraction'],
                    n_tiles_anchor=anchor['n_tiles'],
                ))
                print(f'[{ix}/{len(images)}] {relative} {mode}: reflect={current:.4f} '
                      f'edge={edge["score"]:.4f} anchor={anchor["score"]:.4f}', flush=True)
        except Exception as exc:
            errors.append(dict(image=relative, mode='unknown', reason=f'{type(exc).__name__}: {exc}'))
            if args.fail_fast:
                raise
    with (args.output / 'protocol_per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'errors.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'mode', 'reason'])
        writer.writeheader()
        writer.writerows(errors)
    by_mode = {}
    for mode in modes:
        items = [r for r in rows if r['resolution_mode'] == mode]
        summaries = {
            'edge_vs_reflect_same_crop': ranking_summary(items, 'same_crop_edge'),
            'anchor_vs_reflect_changed_windows': ranking_summary(items, 'full_tile_anchor'),
        }
        if args.grid_phase:
            summaries['even_vs_anchor_grid_phase'] = ranking_summary(
                items, 'full_tile_even', 'full_tile_anchor')
        by_mode[mode] = summaries
    if args.mode == 'both':
        # Paired across dimensions ONLY for identical image identifiers.
        native = {r['relative_path']: r for r in rows if r['resolution_mode'] == 'native'}
        matched = {r['relative_path']: r for r in rows if r['resolution_mode'] == 'matched'}
        common = sorted(native.keys() & matched.keys())
        for strategy in ('legacy_reflect', 'full_tile_anchor'):
            comparison = [{ 'native': native[k][strategy], 'matched': matched[k][strategy] }
                          for k in common]
            by_mode[f'resize_{strategy}'] = ranking_summary(comparison, 'matched', 'native')
    result = {
        'status': 'diagnostic_only_not_proof_of_validity',
        'root_cause_rules': {
            'edge_vs_reflect': 'same crop/pixels; isolates PADDING RULE sensitivity',
            'anchor_vs_reflect': 'padding removed but window locations and overlap changed; NOT a causal estimate of padding',
            'even_vs_anchor': 'same image, full windows, different grid positions; grid-phase sensitivity',
            'native_vs_matched': 'same source field of view, different spatial scale; downsampling may change SE',
        },
        'settings': dict(tile_size=args.tile_size, stride=args.stride,
                         levels=args.levels, order=args.order, backend=args.backend,
                         residuals=args.residuals, matched_long_edge=args.matched_long_edge),
        'images_discovered': len(images), 'image_mode_rows': len(rows),
        'errors': len(errors), 'by_mode': by_mode,
    }
    (args.output / 'protocol_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['native','matched','both'], default='native')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--matched-long-edge', type=int, default=2048)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--order', type=int, choices=[1,3,5], default=3)
    parser.add_argument('--backend', choices=['space','freq'], default='space')
    parser.add_argument('--residuals', choices=['none','lowpass','highpass','both'], default='none')
    parser.add_argument('--grid-phase', action='store_true', help='Additional full-tile grid-phase control')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error('Input folder does not exist')
    if args.tile_size < 64 or args.matched_long_edge < 64:
        parser.error('Tile/longest-edge sizes must be >=64')
    if not 1 <= args.stride <= args.tile_size:
        parser.error('stride must be between 1 and tile-size')
    if args.limit < 0 or args.levels < 1:
        parser.error('limit must be nonnegative; levels must be >=1')
    return args


if __name__ == '__main__':
    results = audit(parse_args())
    print(json.dumps(results, indent=2))
