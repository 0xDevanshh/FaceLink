import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _never_upload_for_real(monkeypatch):
    """Tests must not depend on — or be slowed down by — whatever the
    developer's local `.env` happens to have `ALLOW_UPLOAD_HOST` set to.

    `Settings` reads `.env` unconditionally (see `config.py`), so a real dev
    setup with uploads enabled for actual scans would otherwise make every
    test that reaches the search stage silently upload real image bytes to a
    public host over the network. A test that specifically wants to exercise
    the upload path opts in explicitly with its own `monkeypatch`.
    """
    from facechain.config import settings
    monkeypatch.setattr(settings, "allow_upload_host", False)
