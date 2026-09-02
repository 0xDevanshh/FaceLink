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
