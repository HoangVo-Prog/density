"""Small-array and CLI regression tests for multifeature_entropy.py.

Run with: PYTHONPATH=. python -m pytest -q test_multifeature_entropy.py
"""
from pathlib import Path
from types import SimpleNamespace
import csv

import numpy as np
from PIL import Image

from multifeature_entropy import (
    _hist_from_fft, _hist_from_pairs, _seed_for, discover_images,
    local_entropies, main, shannon_counts, spatial_entropy,
)


def test_exact_fft_matches_exhaustive_distances():
    rng = np.random.default_rng(721)
    for shape in [(2, 7), (7, 2), (10, 12), (3, 3), (17, 19)]:
        a = rng.integers(0, 7, size=shape, dtype=np.uint8)
        for q in np.unique(a):
            y, x = np.nonzero(a == q)
            assert np.array_equal(_hist_from_fft(y, x, chunk_rows=3),
                                  _hist_from_pairs(y, x, 200000))


def test_uniform_grayscale_lbp_topology_are_zero():
    for value in (0, 128, 255):
        a = np.full((13, 18), value, dtype=np.uint8)
        assert np.allclose(local_entropies(a), [0.0, 0.0, 0.0], atol=1e-12)


def test_neighborhood_and_lbp_chunk_invariant():
    rng = np.random.default_rng(200)
    a = rng.integers(0, 256, size=(29, 32), dtype=np.uint8)
    assert np.allclose(local_entropies(a, rows_per_chunk=2),
                       local_entropies(a, rows_per_chunk=29), atol=1e-12)


def test_spatial_exact_sampling_metadata():
    arr = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=np.uint8)
    kw = dict(max_exact_pairs=10000, sample_pairs=100,
              fft_workers=1, fft_chunk_rows=8, denominator="all256",
              seed_for_level=lambda level: 51)
    a = spatial_entropy(arr, mode="auto", **kw)
    b = spatial_entropy(arr, mode="fft", **kw)
    c = spatial_entropy(arr, mode="sample", **kw)
    assert np.isclose(a.value, b.value, atol=1e-12)
    assert a.sampled_levels == 0 and b.sampled_levels == 0
    assert c.sampled_levels == 2 and c.sampled_pairs == 200
    assert a.denominator == 256


def test_cli_both_recursion_and_csv(tmp_path: Path):
    inp = tmp_path / 'imgs'
    (inp / 'nested').mkdir(parents=True)
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(16, 18, 3), dtype=np.uint8)
    Image.fromarray(arr).save(inp / 'nested' / 'photo.JPEG', quality=95)
    Image.fromarray(arr).save(inp / 'other.JPG', quality=95)
    assert len(discover_images(inp, False)) == 1
    assert len(discover_images(inp, True)) == 2
    output = tmp_path / 'results.csv'
    rc = main(['--input',str(inp), '--output',str(output), '--recursive',
               '--color-mode','both', '--spatial-mode','fft', '--progress-classes','0'])
    assert rc == 0
    with output.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r['color_mode'] for r in rows} == {'gray','rgb'}
    for row in rows:
        assert row['status'] == 'ok'
        assert float(row['H_total']) > 0
        assert float(row['H_spatial']) > 0
        if row['color_mode'] == 'rgb':
            assert np.isclose(float(row['H_total']),
                              sum(float(row['H_total_'+band]) for band in 'RGB'))
    assert output.with_suffix('.meta.json').exists()
