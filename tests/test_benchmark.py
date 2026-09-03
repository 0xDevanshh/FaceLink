"""Benchmark module smoke tests — offline, no real model needed."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


def _fake_embed(path, backend_name=None):
    """Return a deterministic embedding based on the filename."""
    seed = sum(ord(c) for c in str(path))
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def patch_embed(monkeypatch):
    monkeypatch.setattr("facechain.benchmark._embed", _fake_embed)
    yield


class TestRunBenchmark:
    def test_requires_at_least_two_genuine(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import run_benchmark
        # Create dummy files
        a = tmp_path / "a.jpg"; a.write_bytes(b"x")
        b = tmp_path / "b.jpg"; b.write_bytes(b"y")
        c = tmp_path / "c.jpg"; c.write_bytes(b"z")

        rc = run_benchmark([str(a)], [str(b), str(c)])
        assert rc == 1

    def test_requires_at_least_two_impostor(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import run_benchmark
        a = tmp_path / "a.jpg"; a.write_bytes(b"x")
        b = tmp_path / "b.jpg"; b.write_bytes(b"y")
        c = tmp_path / "c.jpg"; c.write_bytes(b"z")

        rc = run_benchmark([str(a), str(b)], [str(c)])
        assert rc == 1

    def test_happy_path_exits_zero(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import run_benchmark
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"; p.write_bytes(b"x" * (i + 1))
            imgs.append(str(p))

        rc = run_benchmark(imgs[:2], imgs[2:])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GENUINE PAIRS" in out
        assert "IMPOSTOR PAIRS" in out
        assert "THRESHOLD SWEEP" in out

    def test_verbose_shows_cross_class(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import run_benchmark
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"; p.write_bytes(b"x" * (i + 1))
            imgs.append(str(p))

        run_benchmark(imgs[:2], imgs[2:], verbose=True)
        out = capsys.readouterr().out
        assert "CROSS-CLASS" in out

    def test_missing_file_skipped_gracefully(self, tmp_path, capsys):
        """A missing file should not crash the benchmark."""
        from facechain.benchmark import run_benchmark

        real = tmp_path / "real.jpg"; real.write_bytes(b"x")
        # Patch _embed to return None for nonexistent and a vector for real
        def _embed_stub(path, backend_name=None):
            if "nonexistent" in str(path):
                return None
            return _fake_embed(path)

        with patch("facechain.benchmark._embed", side_effect=_embed_stub):
            rc = run_benchmark(
                [str(real), str(tmp_path / "nonexistent.jpg")],
                [str(real), str(real)],
            )
        # Should fail because after skipping nonexistent we have < 2 genuine embeds
        # OR succeed if real is counted twice. Either way it must not raise.
        assert rc in (0, 1)

    def test_main_entrypoint(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import main
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"; p.write_bytes(b"data")
            imgs.append(str(p))
        rc = main(["--genuine", imgs[0], imgs[1], "--impostor", imgs[2], imgs[3]])
        assert rc == 0


class TestCalibrate:
    def test_below_min_pairs_is_reported_as_insufficient(self):
        from facechain.benchmark import calibrate

        result = calibrate(genuine_scores=[0.9, 0.85], impostor_scores=[0.1, 0.15],
                           min_pairs=50, default_threshold=0.38)
        assert result.status == "CALIBRATION_INSUFFICIENT"
        assert result.suggested_threshold == 0.38  # falls back to the default, not a guess
        assert "50" in result.note

    def test_at_or_above_min_pairs_is_calibrated(self):
        from facechain.benchmark import calibrate

        genuine = [0.85] * 10  # 10 * 10 impostor => enough pairs at min_pairs=10
        impostor = [0.10] * 10
        result = calibrate(genuine, impostor, min_pairs=10, default_threshold=0.38)
        assert result.status == "CALIBRATED"

    def test_well_separated_distributions_find_a_threshold_between_them(self):
        from facechain.benchmark import calibrate

        genuine = [0.90] * 60
        impostor = [0.10] * 60
        result = calibrate(genuine, impostor, min_pairs=50)
        assert result.status == "CALIBRATED"
        assert 0.10 < result.suggested_threshold < 0.90
        assert result.far_at_suggested_pct == pytest.approx(0.0)
        assert result.frr_at_suggested_pct == pytest.approx(0.0)

    def test_sweep_covers_the_documented_threshold_range(self):
        from facechain.benchmark import calibrate

        result = calibrate([0.8] * 55, [0.2] * 55, min_pairs=50)
        thresholds = [row.threshold for row in result.sweep]
        assert min(thresholds) == pytest.approx(0.20)
        assert max(thresholds) == pytest.approx(0.95)

    def test_empty_inputs_are_insufficient_not_a_crash(self):
        from facechain.benchmark import calibrate

        result = calibrate([], [], min_pairs=50)
        assert result.status == "CALIBRATION_INSUFFICIENT"
        assert result.n_genuine_pairs == 0
        assert result.n_impostor_pairs == 0

    def test_insufficient_status_still_returns_a_full_sweep(self):
        """The sweep table is still informative even when the verdict is
        insufficient — only the *recommendation* is withheld."""
        from facechain.benchmark import calibrate

        result = calibrate([0.9], [0.1], min_pairs=50)
        assert result.status == "CALIBRATION_INSUFFICIENT"
        assert len(result.sweep) == 16  # 0.20 .. 0.95 in 0.05 steps


class TestRunBenchmarkCalibrationStatus:
    def test_small_sample_reports_calibration_insufficient(self, patch_embed, tmp_path, capsys):
        from facechain.benchmark import run_benchmark
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"; p.write_bytes(b"x" * (i + 1))
            imgs.append(str(p))

        rc = run_benchmark(imgs[:2], imgs[2:])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CALIBRATION_INSUFFICIENT" in out


class TestLoadCalibrationStatus:
    def test_no_path_returns_default(self):
        from facechain.benchmark import load_calibration_status
        status, note = load_calibration_status("")
        assert status == "DEFAULT"
        assert "default" in note.lower()

    def test_missing_file_falls_back_to_default_not_a_crash(self, tmp_path):
        from facechain.benchmark import load_calibration_status
        status, note = load_calibration_status(str(tmp_path / "does_not_exist.json"))
        assert status == "DEFAULT"
        assert "unreadable" in note.lower()

    def test_malformed_json_falls_back_to_default_not_a_crash(self, tmp_path):
        from facechain.benchmark import load_calibration_status
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        status, note = load_calibration_status(str(p))
        assert status == "DEFAULT"

    def test_a_valid_calibrated_file_is_read_through(self, tmp_path):
        from facechain.benchmark import load_calibration_status
        import json
        p = tmp_path / "calibration.json"
        p.write_text(json.dumps({"status": "CALIBRATED", "note": "60 genuine / 60 impostor pairs"}))
        status, note = load_calibration_status(str(p))
        assert status == "CALIBRATED"
        assert "60" in note

    def test_run_benchmark_out_writes_a_loadable_calibration_file(self, patch_embed, tmp_path):
        from facechain.benchmark import run_benchmark, load_calibration_status
        imgs = []
        for i in range(4):
            p = tmp_path / f"img{i}.jpg"; p.write_bytes(b"x" * (i + 1))
            imgs.append(str(p))
        out = tmp_path / "calibration.json"

        rc = run_benchmark(imgs[:2], imgs[2:], out_path=str(out))
        assert rc == 0
        assert out.exists()
        status, _note = load_calibration_status(str(out))
        # 1 genuine pair / 1 impostor-cross-pair set is below min_pairs, so the
        # round trip must faithfully report insufficiency, not silently upgrade it.
        assert status == "CALIBRATION_INSUFFICIENT"


class TestPairwiseStats:
    def test_empty_returns_none_mean(self):
        from facechain.benchmark import _stats
        s = _stats([])
        assert s["n"] == 0
        assert s["mean"] is None

    def test_single_value(self):
        from facechain.benchmark import _stats
        s = _stats([0.75])
        assert s["n"] == 1
        assert s["mean"] == pytest.approx(0.75)

    def test_multiple_values(self):
        from facechain.benchmark import _stats
        s = _stats([0.4, 0.6, 0.8])
        assert s["n"] == 3
        assert s["mean"] == pytest.approx(0.6, abs=1e-4)
