#!/usr/bin/env python3
"""Multi-Feature Entropy (Luo et al., Remote Sensing 2022, 14, 4068).

Four terms (paper Eq. 1, 3, 4, 5), total (Eq. 7), RGB sum (Eq. 8).
https://doi.org/10.3390/rs14164068

IMPORTANT:
  * Exact spatial modes use the discrete distribution of Euclidean distances
    between all unordered pairs of equal-intensity pixels (no arbitrary bins).
    Squared integer Euclidean distance is an exact distance-category key.
  * FFT mode computes EXACT pair-distance counts (roundoff verified against
    n * (n - 1) / 2); it avoids constructing the quadratic list of pairs.
  * sample mode estimates spatial entropy using sampled pairs; its plug-in
    histogram entropy is generally biased downward and is NOT interchangeable
    with exact scores.
  * Paper does not specify every low-level implementation detail; see README.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from scipy import fft

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
LOG2_COUNTS = np.zeros(9, dtype=np.float64)
LOG2_COUNTS[1:] = np.log2(np.arange(1, 9, dtype=np.float64))


def shannon_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts)
    counts = counts[counts > 0].astype(np.float64, copy=False)
    if not len(counts):
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def local_entropies(channel: np.ndarray, rows_per_chunk: int = 256, lbp_compare: str = "ge") -> tuple[float, float, float]:
    """Calculate H_gray, H_con (LBP), H_nei exactly on 8-bit band.

    H_nei counts grey-value multiplicities among the EIGHT NEIGHBORS,
    NOT the nine pixels including the center. Edge padding repeats edge pixels.
    """
    if channel.ndim != 2 or channel.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 image band")
    h, w = channel.shape
    if h == 0 or w == 0:
        raise ValueError("Empty image")
    h_gray = shannon_counts(np.bincount(channel.ravel(), minlength=256))
    padded = np.pad(channel, ((1, 1), (1, 1)), mode="edge")
    lbp_hist = np.zeros(256, dtype=np.int64)
    sum_log_counts = 0.0

    for y0 in range(0, h, rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, h)
        center = padded[y0 + 1:y1 + 1, 1:w + 1]
        nbrs = [padded[y0 + 1 + dy:y1 + 1 + dy, 1 + dx:w + 1 + dx]
                for dy, dx in NEIGHBORS]
        lbp = np.zeros(center.shape, dtype=np.uint8)
        for bit, nb in enumerate(nbrs):
            condition = (nb >= center) if lbp_compare == "ge" else (nb > center)
            lbp |= condition.astype(np.uint8) << bit
        lbp_hist += np.bincount(lbp.ravel(), minlength=256)

        # Equivalent to computing each neighborhood's categorical histogram:
        #   H = -sum_v (c_v/8) log2(c_v/8)
        #     = 3 - (1/8) sum_{j=1..8} log2(c_{value_at_neighbor_j}).
        equal_multiplicity = [np.ones(center.shape, dtype=np.uint8) for _ in range(8)]
        for i in range(8):
            for j in range(i + 1, 8):
                same = (nbrs[i] == nbrs[j]).astype(np.uint8)
                equal_multiplicity[i] += same
                equal_multiplicity[j] += same
        for mult in equal_multiplicity:
            sum_log_counts += float(np.sum(LOG2_COUNTS[mult], dtype=np.float64))

    h_con = shannon_counts(lbp_hist)
    h_nei = max(0.0, 3.0 - sum_log_counts / (8.0 * h * w))
    return h_gray, h_con, h_nei


def _hist_from_pairs(y: np.ndarray, x: np.ndarray, max_pairs: int) -> np.ndarray:
    n = len(y)
    n_pairs = n * (n - 1) // 2
    if n_pairs > max_pairs:
        raise ValueError(f"Exact pair mode refused {n_pairs:,} pairs > --max-exact-pairs={max_pairs:,}. "
                         "Use --spatial-mode fft (exact) or sample (approximate).")
    if n < 2:
        return np.array([], dtype=np.int64)
    hist = np.zeros((int(np.ptp(y)) ** 2 + int(np.ptp(x)) ** 2 + 1), dtype=np.int64)
    y = y.astype(np.int64, copy=False)
    x = x.astype(np.int64, copy=False)
    for i in range(n - 1):
        d2 = (y[i+1:] - y[i]) ** 2 + (x[i+1:] - x[i]) ** 2
        # bincount is cheaper than repeated np.add.at on reasonably sized batches.
        counts = np.bincount(d2)
        hist[:len(counts)] += counts
    assert int(hist.sum()) == n_pairs
    return hist


def _append_corr_hist(hist: np.ndarray, lag: np.ndarray, dy0: int, dx2: np.ndarray) -> None:
    """Accumulate exact autocorrelation counts over a block of lags."""
    rounded = np.rint(lag)
    idx_r, idx_c = np.nonzero(rounded > 0)
    if idx_r.size == 0:
        return
    keys = (idx_r.astype(np.int64) + dy0) ** 2 + dx2[idx_c]
    freq = rounded[idx_r, idx_c]
    add = np.bincount(keys, weights=freq)
    # Integer sums are safely represented as float64 for 4K images (<2**53).
    hist[:len(add)] += np.rint(add).astype(np.int64)


def _hist_from_fft(y: np.ndarray, x: np.ndarray, workers: int = 1, chunk_rows: int = 128) -> np.ndarray:
    """Exact Euclidean-distance category histogram via ZERO-PADDED 2D autocorrelation.

    All unordered pairs counted exactly once using (dy>0, any dx) or
    (dy=0, dx>0). Floating FFT counts are rounded; pair count is verified.
    The spatial bounding rectangle is cropped without changing distances.
    """
    n = len(y)
    if n < 2:
        return np.array([], dtype=np.int64)
    ya, yb = int(y.min()), int(y.max())
    xa, xb = int(x.min()), int(x.max())
    h, w = yb - ya + 1, xb - xa + 1
    mask = np.zeros((h, w), dtype=np.float64)
    mask[y - ya, x - xa] = 1.0
    fh, fw = fft.next_fast_len(2*h-1), fft.next_fast_len(2*w-1)
    spectrum = fft.rfftn(mask, s=(fh, fw), workers=workers)
    del mask
    spectrum *= np.conj(spectrum)
    corr = fft.irfftn(spectrum, s=(fh, fw), workers=workers)
    del spectrum

    hist = np.zeros((h-1)**2 + (w-1)**2 + 1, dtype=np.int64)
    # dy=0, dx>0 only; dy>0 includes dx=0 and both signs of dx.
    if w > 1:
        a = np.maximum(np.rint(corr[0, 1:w]), 0).astype(np.int64)
        row0_hist = np.bincount(np.arange(1, w, dtype=np.int64)**2, weights=a)
        hist[:len(row0_hist)] += np.rint(row0_hist).astype(np.int64)
    dx2_plus = np.arange(w, dtype=np.int64) ** 2
    dx2_minus = np.arange(w-1, 0, -1, dtype=np.int64) ** 2
    for start in range(1, h, chunk_rows):
        stop = min(start + chunk_rows, h)
        _append_corr_hist(hist, corr[start:stop, :w], start, dx2_plus)
        if w > 1:
            _append_corr_hist(hist, corr[start:stop, fw-(w-1):fw], start, dx2_minus)
    del corr
    expected = n * (n-1)//2
    actual = int(hist.sum())
    if actual != expected:
        raise RuntimeError(f"FFT numerical count mismatch: expected={expected}, actual={actual}; "
                           "check hardware/FFT implementation; no silent correction performed")
    return hist


def _hist_from_sample(y: np.ndarray, x: np.ndarray, pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Sample distinct pixel pairs, with replacement, uniformly from unordered pairs.

    Returns nonzero histogram frequencies only; categories can be omitted because
    only Shannon entropy is consumed. Beware negative plug-in entropy bias.
    """
    n = len(y)
    if n < 2:
        return np.array([], dtype=np.int64)
    chunks = []
    drawn = 0
    while drawn < pairs:
        n_draw = min(200_000, pairs - drawn)
        a = rng.integers(0, n, n_draw)
        b = rng.integers(0, n - 1, n_draw)
        b += (b >= a)  # Uniform distinct second point, no rejection
        d2 = ((y[a].astype(np.int64) - y[b].astype(np.int64))**2
              + (x[a].astype(np.int64) - x[b].astype(np.int64))**2)
        chunks.append(d2)
        drawn += n_draw
    return np.unique(np.concatenate(chunks), return_counts=True)[1]


@dataclass
class SpatialReport:
    value: float | None
    exact_levels: int
    sampled_levels: int
    fft_levels: int
    pair_levels: int
    total_possible_pairs: int
    sampled_pairs: int
    occupied_levels: int
    denominator: int


def spatial_entropy(channel: np.ndarray, *, mode: str, max_exact_pairs: int,
                    sample_pairs: int, fft_workers: int, fft_chunk_rows: int,
                    denominator: str, seed_for_level: Callable[[int], int],
                    progress: Callable[[int, int], None] | None = None) -> SpatialReport:
    """H_spa: mean of conditional same-grayscale distance-histogram entropies."""
    levels, frequencies = np.unique(channel, return_counts=True)
    counts_per_level = dict(zip(levels.tolist(), frequencies.tolist()))
    number_occupied = len(counts_per_level)
    denom = 256 if denominator == "all256" else number_occupied
    pairs_total = sum(n*(n-1)//2 for n in counts_per_level.values())
    if mode == "off":
        return SpatialReport(None, 0, 0, 0, 0, pairs_total, 0, number_occupied, denom)
    total_h = 0.0
    exact_levels = sampled_levels = fft_levels = pair_levels = sample_n = 0
    flat = channel.ravel()
    width = channel.shape[1]
    for k, level in enumerate(levels.tolist()):
        n = counts_per_level[level]
        if n < 2:
            if progress is not None:
                progress(k+1, len(levels))
            continue
        n_pairs = n*(n-1)//2
        # The user-facing auto mode is exact: direct for small classes,
        # zero-padded FFT for large classes. It NEVER silently samples.
        resolved = ("pairs" if n_pairs <= max_exact_pairs else "fft") if mode == "auto" else mode
        if resolved == "pairs" and n_pairs > max_exact_pairs:
            raise ValueError(f"Intensity level {level}: {n_pairs:,} pairs above --max-exact-pairs. "
                             "Choose --spatial-mode auto / fft / sample.")
        indices = np.flatnonzero(flat == level)
        y, x = np.divmod(indices, width)
        del indices
        if resolved == "pairs":
            dist_hist = _hist_from_pairs(y, x, max_exact_pairs)
            pair_levels += 1
            exact_levels += 1
        elif resolved == "fft":
            dist_hist = _hist_from_fft(y, x, workers=fft_workers, chunk_rows=fft_chunk_rows)
            fft_levels += 1
            exact_levels += 1
        elif resolved == "sample":
            dist_hist = _hist_from_sample(y, x, sample_pairs,
                                          rng=np.random.default_rng(seed_for_level(level)))
            sampled_levels += 1
            sample_n += sample_pairs
        else:
            raise ValueError(resolved)
        total_h += shannon_counts(dist_hist)
        if progress is not None:
            progress(k+1, len(levels))
    return SpatialReport(total_h / denom, exact_levels, sampled_levels,
                         fft_levels, pair_levels, pairs_total, sample_n,
                         number_occupied, denom)


def _seed_for(seed: int, image_id: str, band_name: str, level: int) -> int:
    b = f"{seed}|{image_id}|{band_name}|{level}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little")


def measure_band(channel: np.ndarray, *, args: argparse.Namespace, image_id: str,
                 band_name: str) -> dict:
    start = time.perf_counter()
    g, con, nei = local_entropies(channel, rows_per_chunk=args.local_chunk_rows,
                                 lbp_compare=args.lbp_compare)
    def progress(i: int, n: int) -> None:
        if args.progress_classes and (i % args.progress_classes == 0 or i == n):
            print(f"    {image_id} [{band_name}] spatial levels {i}/{n}", file=sys.stderr, flush=True)
    s = spatial_entropy(channel, mode=args.spatial_mode,
                        max_exact_pairs=args.max_exact_pairs,
                        sample_pairs=args.sample_pairs,
                        fft_workers=args.fft_workers,
                        fft_chunk_rows=args.fft_chunk_rows,
                        denominator=args.spatial_denominator,
                        seed_for_level=lambda level: _seed_for(args.seed, image_id, band_name, level),
                        progress=progress if args.progress_classes else None)
    weights = args.weights
    total = (weights[0]*g + weights[1]*con + weights[2]*nei + weights[3]*s.value
             if s.value is not None else None)
    return {
        "H_gray": g, "H_contrast": con, "H_neighborhood": nei,
        "H_spatial": s.value, "H_total": total,
        "H_partial_without_spatial": weights[0]*g+weights[1]*con+weights[2]*nei,
        "spatial_exact_levels": s.exact_levels,
        "spatial_sampled_levels": s.sampled_levels,
        "spatial_fft_levels": s.fft_levels,
        "spatial_pair_levels": s.pair_levels,
        "spatial_total_possible_pairs": s.total_possible_pairs,
        "spatial_sampled_pairs": s.sampled_pairs,
        "spatial_occupied_levels": s.occupied_levels,
        "spatial_denominator_value": s.denominator,
        "band_seconds": time.perf_counter()-start,
    }


def discover_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted((p for p in iterator if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
                  key=lambda p: str(p.relative_to(root)).casefold())


def analyze_image(path: Path, root: Path, args: argparse.Namespace) -> list[dict]:
    t0 = time.perf_counter()
    image_id = str(path.relative_to(root))
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    common = {"image": image_id, "width": w, "height": h, "megapixels": w*h/1e6,
              "spatial_mode": args.spatial_mode, "spatial_denominator": args.spatial_denominator,
              "lbp_compare": args.lbp_compare, "weights": ",".join(map(str,args.weights)),
              "seed": args.seed, "sample_pairs_per_graylevel": args.sample_pairs,
              "status": "ok", "error": ""}
    rows: list[dict] = []
    if args.color_mode in {"gray", "both"}:
        gray = np.asarray(Image.fromarray(rgb, "RGB").convert("L"), dtype=np.uint8)
        print(f"  grayscale band: {image_id}", file=sys.stderr, flush=True)
        report = measure_band(gray, args=args, image_id=image_id, band_name="L")
        rows.append({**common, "color_mode": "gray", **report})
    if args.color_mode in {"rgb", "both"}:
        per_band = {}
        for index, name in enumerate(("R", "G", "B")):
            print(f"  RGB band {name}: {image_id}", file=sys.stderr, flush=True)
            per_band[name] = measure_band(rgb[:, :, index], args=args,
                                          image_id=image_id, band_name=name)
        aggreg = {}
        for field in ("H_gray", "H_contrast", "H_neighborhood", "H_spatial", "H_total",
                      "H_partial_without_spatial", "spatial_exact_levels", "spatial_sampled_levels",
                      "spatial_fft_levels", "spatial_pair_levels", "spatial_total_possible_pairs",
                      "spatial_sampled_pairs", "spatial_occupied_levels", "spatial_denominator_value",
                      "band_seconds"):
            v = [per_band[n][field] for n in ("R", "G", "B")]
            aggreg[field] = (sum(v) if all(x is not None for x in v) else None)
        # RGB follows Eq. (8): 3 single-band totals summed, NOT averaged.
        for name in ("R", "G", "B"):
            for field in ("H_gray", "H_contrast", "H_neighborhood", "H_spatial", "H_total"):
                aggreg[f"{field}_{name}"] = per_band[name][field]
        rows.append({**common, "color_mode": "rgb", **aggreg})
    elapsed = time.perf_counter()-t0
    for row in rows:
        row["image_seconds_total"] = elapsed
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Multi-Feature Entropy of JPEG images (Luo et al. 2022 Eq. 1–8).")
    ap.add_argument("--input", type=Path, help="Input image folder")
    ap.add_argument("--output", default=Path("multifeature_entropy.csv"), type=Path)
    ap.add_argument("--recursive", action="store_true", help="Include all subdirectories")
    ap.add_argument("--color-mode", choices=["gray", "rgb", "both"], default="gray",
                    help="gray mirrors paper experiments; rgb sums separate R/G/B per Eq. 8")
    ap.add_argument("--spatial-mode", choices=["auto", "pairs", "fft", "sample", "off"],
                    default="auto", help="auto=exact direct/FFT; sample=APPROXIMATE; off=partial only")
    ap.add_argument("--max-exact-pairs", type=int, default=1_000_000,
                    help="Per gray-level threshold for direct pair counting (auto) / cap (pairs)")
    ap.add_argument("--sample-pairs", type=int, default=200_000,
                    help="Random pair draws per gray level in sample mode (biased plug-in estimate)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spatial-denominator", choices=["all256", "occupied"], default="all256",
                    help="all256 uses Q=256 grayscale levels (Eq. 5); occupied only those present")
    ap.add_argument("--lbp-compare", choices=["ge", "gt"], default="ge",
                    help="ge follows paper Eq. 2 f(x)>=0; gt follows prose 'greater than'")
    ap.add_argument("--weights", type=float, nargs=4, default=[1., 1., 1., 1.],
                    metavar=("GRAY", "CONTRAST", "NEIGHBOR", "SPATIAL"),
                    help="Paper Eq. 7 weights; defaults all 1")
    ap.add_argument("--local-chunk-rows", type=int, default=128)
    ap.add_argument("--fft-chunk-rows", type=int, default=128)
    ap.add_argument("--fft-workers", type=int, default=1,
                    help="SciPy FFT CPU workers; 1 is memory-friendlier")
    ap.add_argument("--progress-classes", type=int, default=16,
                    help="Progress print every N gray levels; 0 disables")
    ap.add_argument("--on-error", choices=["continue", "raise"], default="continue")
    ap.add_argument("--self-test", action="store_true", help="Run built-in small-image unit tests and exit")
    args = ap.parse_args(argv)
    if args.sample_pairs < 1 or args.max_exact_pairs < 1:
        ap.error("--sample-pairs and --max-exact-pairs must be positive")
    if args.local_chunk_rows < 1 or args.fft_chunk_rows < 1 or args.fft_workers < 1:
        ap.error("chunk rows and fft-workers must be positive")
    if any(not math.isfinite(w) for w in args.weights):
        ap.error("weights must be finite")
    return args


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write after each image: interrupted batches still leave a complete CSV.
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _self_test() -> None:
    import itertools
    from numpy.testing import assert_allclose
    sample = np.array([[0, 3, 0, 3], [3, 0, 3, 0], [0, 0, 3, 3]], dtype=np.uint8)
    # All 3 spatial exact methods must agree (auto uses pairs for tiny images).
    for level in (0, 3):
        y, x = np.nonzero(sample == level)
        pair_h = _hist_from_pairs(y, x, 100000)
        fft_h = _hist_from_fft(y, x)
        assert np.array_equal(pair_h, fft_h), (level, pair_h, fft_h)
    g, con, nei = local_entropies(np.zeros((4, 5), dtype=np.uint8))
    assert_allclose((g, con, nei), (0., 0., 0.), atol=1e-12)
    g2, con2, nei2 = local_entropies(sample, rows_per_chunk=1)
    g3, con3, nei3 = local_entropies(sample, rows_per_chunk=128)
    assert_allclose((g2, con2, nei2), (g3, con3, nei3), atol=1e-12)
    # Brute force validation: 8 neighbors' exact categorical entropy.
    padded = np.pad(sample, 1, mode="edge")
    vals = []
    for y, x in itertools.product(range(sample.shape[0]), range(sample.shape[1])):
        ns = [padded[y+1+dy,x+1+dx] for dy,dx in NEIGHBORS]
        vals.append(shannon_counts(np.unique(ns, return_counts=True)[1]))
    assert_allclose(nei2, np.mean(vals), atol=1e-12)
    for denom in ("all256", "occupied"):
        pa = spatial_entropy(sample, mode="pairs", max_exact_pairs=10000,
                             sample_pairs=10, fft_workers=1, fft_chunk_rows=8,
                             denominator=denom, seed_for_level=lambda x: 42)
        fa = spatial_entropy(sample, mode="fft", max_exact_pairs=10000,
                             sample_pairs=10, fft_workers=1, fft_chunk_rows=8,
                             denominator=denom, seed_for_level=lambda x: 42)
        assert_allclose(pa.value, fa.value, atol=1e-12)
    # Random pairs on 4 small arrays, and stripe layouts to test negative dx.
    rng = np.random.default_rng(92)
    for shape in ((8, 12), (12, 8), (1, 15), (15, 1), (15, 19)):
        a = rng.integers(0, 5, shape, dtype=np.uint8)
        for level in np.unique(a):
            y, x = np.nonzero(a == level)
            assert np.array_equal(_hist_from_pairs(y, x, 100000), _hist_from_fft(y, x))
    print("PASS: grayscale, LBP, neighborhood, spatial exact pairs vs FFT, RGB-ready functions")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.input is None or not args.input.is_dir():
        print(f"Not an input directory: {args.input}", file=sys.stderr)
        return 2
    paths = discover_images(args.input, args.recursive)
    if not paths:
        print("No .jpg/.jpeg files found (case-insensitive).", file=sys.stderr)
        return 2
    rows = []
    issues = []
    t0 = time.perf_counter()
    print(f"Found {len(paths)} JPEG files; color={args.color_mode}, spatial={args.spatial_mode}",
          file=sys.stderr, flush=True)
    for index, path in enumerate(paths, 1):
        name = str(path.relative_to(args.input))
        print(f"[{index}/{len(paths)}] {name}", file=sys.stderr, flush=True)
        try:
            rows.extend(analyze_image(path, args.input, args))
        except Exception as exc:
            if args.on_error == "raise":
                raise
            issues.append({"image": name, "error_type": type(exc).__name__, "error": str(exc)})
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rows.append({"image": name, "status": "error", "error": str(exc)})
        _write_csv(args.output, rows)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "paper": "Luo et al. (2022), Remote Sensing 14(16):4068, doi:10.3390/rs14164068",
        "script": "multifeature_entropy.py", "options": {
            k: (str(v) if isinstance(v, Path) else v) for k,v in vars(args).items()
        }, "files_found": len(paths), "failures": issues, "elapsed_seconds": time.perf_counter()-t0,
        "spatial_exact_by_default": args.spatial_mode in {"auto", "fft", "pairs"},
        "important": "FFT and pairs exact; sample is a biased plug-in estimator. Distance bins use integer squared Euclidean distance. all256 vs occupied is an explicit Eq5 convention.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results: {args.output}; metadata: {meta_path}; errors: {len(issues)}", file=sys.stderr)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
