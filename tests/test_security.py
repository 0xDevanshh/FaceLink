"""Security layer tests: SSRF, path traversal, log scrubbing.

All offline — no network required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from facechain.security.ssrf import SSRFViolation, validate_url, safe_url_or_none
from facechain.security.paths import PathTraversalError, safe_case_id, safe_evidence_path
from facechain.security.scrubber import scrub


# ---- SSRF ----------------------------------------------------------------

class TestSSRF:
    def test_rejects_localhost_ip(self):
        with pytest.raises(SSRFViolation, match="blocked address"):
            validate_url("http://127.0.0.1/secret")

    def test_rejects_localhost_name(self):
        with pytest.raises(SSRFViolation):
            validate_url("http://localhost/secret")

    def test_rejects_ipv6_loopback(self):
        with pytest.raises(SSRFViolation):
            validate_url("http://[::1]/secret")

    def test_rejects_private_class_a(self):
        with pytest.raises(SSRFViolation, match="blocked address"):
            validate_url("http://10.0.0.1/internal")

    def test_rejects_private_class_b(self):
        with pytest.raises(SSRFViolation, match="blocked address"):
            validate_url("http://172.16.0.1/internal")

    def test_rejects_private_class_c(self):
        with pytest.raises(SSRFViolation, match="blocked address"):
            validate_url("http://192.168.1.1/router")

    def test_rejects_link_local(self):
        with pytest.raises(SSRFViolation, match="blocked address"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_file_scheme(self):
        with pytest.raises(SSRFViolation, match="scheme"):
            validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(SSRFViolation, match="scheme"):
            validate_url("ftp://example.com/file")

    def test_rejects_gopher_scheme(self):
        with pytest.raises(SSRFViolation, match="scheme"):
            validate_url("gopher://example.com/")

    def test_rejects_no_host(self):
        with pytest.raises(SSRFViolation, match="host"):
            validate_url("https:///path")

    def test_accepts_public_https(self):
        # Mock DNS resolution to return a public IP.
        with patch("facechain.security.ssrf.socket.getaddrinfo",
                   return_value=[(2, 1, 0, '', ('93.184.216.34', None))]):
            result = validate_url("https://example.com/image.jpg")
        assert result == "https://example.com/image.jpg"

    def test_accepts_public_http(self):
        with patch("facechain.security.ssrf.socket.getaddrinfo",
                   return_value=[(2, 1, 0, '', ('93.184.216.34', None))]):
            result = validate_url("http://example.com/image.jpg")
        assert result == "http://example.com/image.jpg"

    def test_safe_url_or_none_returns_none_for_private(self):
        assert safe_url_or_none("http://192.168.1.1/") is None

    def test_safe_url_or_none_returns_url_for_public(self):
        with patch("facechain.security.ssrf.socket.getaddrinfo",
                   return_value=[(2, 1, 0, '', ('93.184.216.34', None))]):
            assert safe_url_or_none("https://example.com/") == "https://example.com/"

    def test_rejects_0_0_0_0(self):
        with pytest.raises(SSRFViolation):
            validate_url("http://0.0.0.0/")

    def test_rejects_metadata_endpoint(self):
        """AWS/GCP metadata endpoint must be rejected."""
        with pytest.raises(SSRFViolation):
            validate_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")


# ---- path traversal ------------------------------------------------------

class TestPathTraversal:
    def test_valid_case_id_accepted(self):
        assert safe_case_id("case_20260901_123456") == "case_20260901_123456"

    def test_dotdot_rejected(self):
        with pytest.raises(PathTraversalError):
            safe_case_id("../etc/passwd")

    def test_slash_rejected(self):
        with pytest.raises(PathTraversalError):
            safe_case_id("case/../../etc")

    def test_backslash_rejected(self):
        with pytest.raises(PathTraversalError):
            safe_case_id("case\\windows\\system32")

    def test_empty_rejected(self):
        with pytest.raises(PathTraversalError):
            safe_case_id("")

    def test_safe_evidence_path_stays_inside_root(self, tmp_path):
        root = tmp_path / "evidence"
        root.mkdir()
        p = safe_evidence_path(root, "case_20260901_000000", "case.json")
        assert str(p).startswith(str(root))

    def test_safe_evidence_path_rejects_traversal_in_filename(self, tmp_path):
        root = tmp_path / "evidence"
        root.mkdir()
        with pytest.raises(PathTraversalError):
            safe_evidence_path(root, "case_20260901_000000", "../../../etc/passwd")

    def test_safe_evidence_path_rejects_bad_case_id(self, tmp_path):
        root = tmp_path / "evidence"
        root.mkdir()
        with pytest.raises(PathTraversalError):
            safe_evidence_path(root, "../../evil", "case.json")


# ---- log scrubber --------------------------------------------------------

class TestLogScrubber:
    def test_scrubs_private_key(self):
        msg = "PRIVATE_KEY=0x" + "a" * 64
        out = scrub(msg)
        assert "a" * 64 not in out
        assert "[REDACTED]" in out

    def test_scrubs_api_key_pattern(self):
        msg = "api_key=sk-abcdefghij1234567890abcdefghijkl"
        out = scrub(msg)
        assert "sk-abcdefghij1234567890abcdefghijkl" not in out

    def test_leaves_normal_log_alone(self):
        msg = "face similarity 0.923 for candidate instagram.com/p/ABC"
        assert scrub(msg) == msg

    def test_scrubs_large_float_array(self):
        arr = "[" + ",".join(["0.1234"] * 60) + "]"
        out = scrub(arr)
        assert "[REDACTED]" in out

    def test_scrubber_install_is_idempotent(self):
        from facechain.security.scrubber import install
        install()
        install()  # second call must not raise


# ---- quality gate --------------------------------------------------------

class TestQualityGate:
    def _make_face(self, bbox=(10, 10, 200, 200), det_score=0.95):
        """Minimal DetectedFace-like object."""
        from facechain.face.detector import DetectedFace
        import numpy as np
        return DetectedFace(bbox=bbox, det_score=det_score,
                            embedding=np.ones(512, dtype=np.float32))

    def test_clean_image_passes(self):
        import numpy as np
        import cv2
        from facechain.face.quality import gate
        # Sharp, well-exposed synthetic face image.
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        cv2.circle(img, (320, 320), 150, (200, 180, 160), -1)  # adds structure/variance
        # Add texture so Laplacian isn't zero.
        noise = np.random.RandomState(0).randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        face = self._make_face()
        report = gate(img, [face])
        assert report.passed

    def test_too_small_face_rejected(self):
        import numpy as np
        from facechain.face.quality import gate, QualityError
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        noise = np.random.RandomState(1).randint(0, 80, img.shape, dtype=np.uint8)
        img = img + noise
        # Face bbox much smaller than min_face_px=80.
        tiny_face = self._make_face(bbox=(10, 10, 20, 20))
        report = gate(img, [tiny_face])
        assert not report.passed
        assert report.error == QualityError.FACE_TOO_SMALL

    def test_no_face_rejected(self):
        import numpy as np
        from facechain.face.quality import gate, QualityError
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        noise = np.random.RandomState(2).randint(0, 80, img.shape, dtype=np.uint8)
        img = img + noise
        report = gate(img, [])
        assert not report.passed
        assert report.error == QualityError.NO_FACE

    def test_blurry_rejected(self):
        import numpy as np
        import cv2
        from facechain.face.quality import gate, QualityError
        # Heavily blurred = low Laplacian variance.
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        img = cv2.GaussianBlur(img, (99, 99), 50)
        face = self._make_face()
        report = gate(img, [face])
        assert not report.passed
        assert report.error == QualityError.BLURRY

    def test_overexposed_rejected(self):
        import numpy as np
        from facechain.face.quality import gate, QualityError
        img = np.full((640, 640, 3), 252, dtype=np.uint8)  # > 240 mean
        face = self._make_face()
        report = gate(img, [face])
        assert not report.passed
        assert report.error == QualityError.HIGH_EXPOSURE

    def test_underexposed_rejected(self):
        import numpy as np
        from facechain.face.quality import gate, QualityError
        img = np.full((640, 640, 3), 5, dtype=np.uint8)  # < 15 mean
        face = self._make_face()
        report = gate(img, [face])
        assert not report.passed
        assert report.error in (QualityError.LOW_EXPOSURE, QualityError.BLURRY)


# ---- confidence bands ----------------------------------------------------

class TestConfidenceBands:
    def test_strong(self):
        from facechain.config import confidence_band
        assert confidence_band(0.90) == "STRONG"
        assert confidence_band(0.85) == "STRONG"

    def test_moderate(self):
        from facechain.config import confidence_band
        assert confidence_band(0.75) == "MODERATE"
        assert confidence_band(0.70) == "MODERATE"

    def test_weak(self):
        from facechain.config import confidence_band
        assert confidence_band(0.60) == "WEAK"
        assert confidence_band(0.50) == "WEAK"

    def test_insufficient(self):
        from facechain.config import confidence_band
        assert confidence_band(0.30) == "INSUFFICIENT"
        assert confidence_band(0.0) == "INSUFFICIENT"
        assert confidence_band(-0.5) == "INSUFFICIENT"
