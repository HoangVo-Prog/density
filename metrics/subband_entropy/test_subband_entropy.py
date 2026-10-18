"""Basic unit tests; end-to-end test uses a fake pyramid to test CLI/IO only.

It does NOT establish numerical equivalence to pyrtools or official MATLAB.
After installing pyrtools, rerun CLI on small diagnostic images and compare
against original MATLAB before claiming reproduction.
"""

import csv
import sys
import types

import numpy as np
from PIL import Image

from measure_subband_entropy import (
    entropy_of_subband, find_jpegs, inverse_coverage_1d,
    parse_args, run, subband_entropy_rgb, MetricSettings,
)


def test_entropy_constant_and_binary():
    assert entropy_of_subband(np.zeros((64, 64))) == 0.0
    x = np.tile([0.0, 1.0], (64, 32))
    assert np.isclose(entropy_of_subband(x), 1.0)


def test_overlap_pixel_weights_sum_to_area():
    for n, tile, stride in [(10, 6, 4), (1025, 512, 256), (100, 512, 512), (257, 128, 128)]:
        inv = inverse_coverage_1d(n, tile, stride)
        a = 0.0
        for start in range(0, n, stride):
            a += inv[start:min(n, start + tile)].sum()
        assert np.isclose(a, n)


def test_recursive_case_insensitive(tmp_path):
    (tmp_path / 'inner').mkdir()
    for p in ['a.JPG', 'b.JpEg', 'inner/c.jpg', 'inner/no.png']:
        (tmp_path / p).write_bytes(b'fake')
    assert len(find_jpegs(tmp_path, recursive=False)) == 2
    assert len(find_jpegs(tmp_path, recursive=True)) == 3


def test_invalid_stride(tmp_path):
    try:
        parse_args(['-i', str(tmp_path), '-o', str(tmp_path), '--tile-size', '64', '--stride', '65'])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError('Expected invalid stride to fail')


def test_cli_e2e_io_with_stub_not_numeric_validation(tmp_path, monkeypatch):
    class FakePyramid:
        def __init__(self, data, height, order, **kwargs):
            self.pyr_coeffs = {}
            for s in range(height):
                for orientation in range(order + 1):
                    self.pyr_coeffs[(s, orientation)] = data[::2**s, ::2**s] + orientation * 0.01
            self.pyr_coeffs['residual_highpass'] = data - np.mean(data)
            self.pyr_coeffs['residual_lowpass'] = data

    fake_pt = types.ModuleType('pyrtools')
    fake_pt.__version__ = 'test-only'
    fake_pt.pyramids = types.SimpleNamespace(
        SteerablePyramidSpace=FakePyramid,
        SteerablePyramidFreq=FakePyramid,
    )
    monkeypatch.setitem(sys.modules, 'pyrtools', fake_pt)

    im_root = tmp_path / 'images'
    im_root.mkdir()
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, (180, 257, 3), dtype=np.uint8)
    Image.fromarray(array).save(im_root / 'image.JPG')
    output = tmp_path / 'out'
    args = parse_args([
        '--input', str(im_root), '--output', str(output),
        '--mode', 'both', '--aggregation', 'both',
        '--matched-long-edge', '128', '--tile-size', '128',
        '--stride', '64', '--residuals', 'both',
        '--no-hash-files',
    ])
    assert run(args) == 0
    with (output / 'image_density.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    with (output / 'tile_density.csv').open(newline='') as f:
        tiles = list(csv.DictReader(f))
    assert len(rows) == 4
    assert len(tiles) > 0
    assert all(float(row['subband_entropy']) >= 0 for row in rows)
    assert {(r['resolution_mode'], r['aggregation_mode']) for r in rows} == {
        ('native', 'global'), ('native', 'tiled'),
        ('matched', 'global'), ('matched', 'tiled'),
    }
    with (output / 'dataset_density_summary.csv').open(newline='') as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 4
    assert (output / 'run_config.json').exists()
    assert (output / 'errors.csv').exists()
