"""Boundary and measurement-protocol tests; run with pytest.

Ordinary tests use deterministic synthetic score functions to prove geometry,
weighting and isolation of confounds. They do NOT validate the steerable pyramid.
The optional real-pyrtools test is skipped when pyrtools is unavailable.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_subband_entropy as se
import audit_se_protocol as diag


@pytest.mark.parametrize('width,height,size,stride', [
    (512, 512, 512, 512), (513, 513, 512, 512),
    (2250, 1500, 512, 512), (2250, 1500, 512, 256),
    (2048, 1365, 512, 512), (1025, 768, 128, 64),
])
@pytest.mark.parametrize('strategy', diag.STRATEGIES)
def test_every_strategy_has_exact_image_coverage(width, height, size, stride, strategy):
    for length in (width, height):
        locs = diag.starts(length, size, stride, strategy)
        assert locs[0] == 0
        if strategy in ('anchor', 'even'):
            assert locs[-1] == length - size
            assert all(x + size <= length for x in locs)
        inv = diag.fair_axis_weights(length, size, locs)
        total = sum(inv[x:min(length, x+size)].sum() for x in locs)
        assert math.isclose(total, length, rel_tol=1e-12, abs_tol=1e-8)


def test_same_crop_padding_control_keeps_original_pixels():
    rgb = np.arange(80*73*3, dtype=np.uint32).reshape(80, 73, 3).astype('uint8')
    reflected = diag.pad_partial(rgb, 128, 'reflect')
    replicated = diag.pad_partial(rgb, 128, 'edge')
    assert np.array_equal(reflected[:80, :73], rgb)
    assert np.array_equal(replicated[:80, :73], rgb)
    assert not np.array_equal(reflected, replicated)
    assert reflected.shape == replicated.shape == (128, 128, 3)


def test_single_bright_pixel_border_changes_padding_rule_not_crop():
    a = np.zeros((128, 135, 3), dtype='uint8')
    a[:, -1, :] = 255
    image = Image.fromarray(a)
    fun = lambda rgb: float(np.asarray(rgb, dtype=float).mean())
    reflect = diag.measure_strategy(image, 128, 128, 'reflect', fun)
    edge = diag.measure_strategy(image, 128, 128, 'edge', fun)
    assert edge['n_tiles'] == reflect['n_tiles'] == 2
    assert edge['padded_tiles'] == reflect['padded_tiles'] == 1
    assert edge['score'] > reflect['score']
    assert edge['padded_effective_area_fraction'] == pytest.approx(7/135)


def test_anchor_eliminates_padding_and_preserves_mass():
    image = Image.fromarray(np.zeros((1500, 2250, 3), dtype='uint8'))
    for strategy in ('anchor', 'even'):
        result = diag.measure_strategy(image, 512, 512, strategy, lambda rgb: 42.0)
        assert result['padded_tiles'] == 0
        assert result['artificial_pixel_fraction'] == 0
        assert result['score'] == pytest.approx(42.0)
    legacy = diag.measure_strategy(image, 512, 512, 'reflect', lambda rgb: 42.0)
    assert legacy['padded_tiles'] > 0
    assert legacy['score'] == pytest.approx(42.0)


def test_diagnostic_reflect_matches_production_implementation(monkeypatch):
    rng = np.random.default_rng(41)
    im = Image.fromarray(rng.integers(0, 256, (230, 275, 3), dtype=np.uint8))

    def pretend_pyramid(rgb, cfg):
        # A deterministic score intentionally sensitive to mirrored content.
        x = np.asarray(rgb, dtype=np.float64)
        values = [float(x[..., c].mean()) for c in range(3)]
        return {
            'subband_entropy': values[0]*.84 + values[1]*.08 + values[2]*.08,
            'entropy_L': values[0], 'entropy_a': values[1], 'entropy_b': values[2],
            'n_subbands_total': 36,
        }
    monkeypatch.setattr(se, 'subband_entropy_rgb', pretend_pyramid)
    fn = lambda rgb: pretend_pyramid(rgb, None)['subband_entropy']
    for stride in (128, 64):
        a = se.tiled_image_score(im, se.MetricSettings(), 128, stride, None, {})
        b = diag.measure_strategy(im, 128, stride, 'reflect', fn)
        assert a['subband_entropy'] == pytest.approx(b['score'], abs=1e-10)
        assert a['n_tiles'] == b['n_tiles']


def test_anchor_vs_reflect_is_not_isolated_padding_effect():
    # When dimension is divisible by tile size both protocols have identical
    # windows and must agree; otherwise their locations deliberately differ.
    assert diag.starts(1024, 512, 512, 'reflect') == diag.starts(1024, 512, 512, 'anchor')
    assert diag.starts(1025, 512, 512, 'reflect') != diag.starts(1025, 512, 512, 'anchor')
    assert diag.starts(1025, 512, 512, 'anchor')[-1] == 513


def test_native_matched_identity_when_below_cap():
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 255, (128, 250, 3), dtype=np.uint8))
    assert se.matched_resize(image, 2048) is image
    assert se.matched_resize(image, 128).size == (128, 66)


def test_rank_statistics_do_not_falsely_assert_invariance():
    rows = [dict(legacy_reflect=a, same_crop_edge=b)
            for a, b in [(1, 2), (2, 1), (3, 3), (4, 4)]]
    s = diag.ranking_summary(rows, 'same_crop_edge')
    assert s['n'] == 4
    assert s['max_rank_shift'] == 1.0
    assert s['spearman_rho'] < 1.0
    assert diag.ranking_summary(rows[:1], 'same_crop_edge')['spearman_rho'] is None


def test_unsupported_full_tile_case_is_explicit():
    with pytest.raises(ValueError, match='requires'):
        diag.starts(150, 512, 512, 'anchor')
    with pytest.raises(ValueError, match='cover'):
        diag.fair_axis_weights(100, 40, [0, 60])


def test_real_pyramid_constant_and_rgb_weights():
    pytest.importorskip('pyrtools')
    a = np.full((128, 128, 3), 96, dtype='uint8')
    constant = se.subband_entropy_rgb(a, se.MetricSettings())
    assert constant['subband_entropy'] == pytest.approx(0, abs=1e-7)
    rng = np.random.default_rng(0)
    b = rng.integers(0, 255, (128, 128, 3), dtype='uint8')
    result = se.subband_entropy_rgb(b, se.MetricSettings())
    assert result['subband_entropy'] == pytest.approx(
        .84*result['entropy_L']+.08*result['entropy_a']+.08*result['entropy_b'], abs=1e-9)
    assert result['n_subbands_total'] == 36


def test_audit_writes_paired_csv_and_json(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    root = tmp_path / 'images'
    root.mkdir()
    Image.fromarray(rng.integers(0, 255, (140, 150, 3), dtype='uint8')).save(root / 'a.jpg')
    Image.fromarray(rng.integers(0, 255, (145, 160, 3), dtype='uint8')).save(root / 'b.JPEG')
    monkeypatch.setattr(se, '_check_pyrtools', lambda: None)
    def mock_score(rgb, cfg):
        x = np.asarray(rgb, dtype=float)
        v = float(x[..., 0].mean())
        return dict(subband_entropy=v, entropy_L=v, entropy_a=0.0,
                    entropy_b=0.0, n_subbands_total=36)
    monkeypatch.setattr(se, 'subband_entropy_rgb', mock_score)
    args = diag.parse_args(['--input', str(root), '--output', str(tmp_path / 'out'),
                            '--mode', 'both', '--limit', '0', '--tile-size', '64',
                            '--stride', '64', '--matched-long-edge', '100', '--grid-phase'])
    report = diag.audit(args)
    assert report['image_mode_rows'] == 4
    assert report['errors'] == 0
    with (args.output / 'protocol_per_image.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r['resolution_mode'] for r in rows} == {'native', 'matched'}
    assert all(r['full_tile_anchor'] != '' for r in rows)
    assert 'resize_full_tile_anchor' in report['by_mode']
