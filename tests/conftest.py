import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_settings_to_shipped_defaults(monkeypatch):
    """Tests must not depend on whatever the developer's local `.env` happens
    to have these set to — `Settings` reads `.env` unconditionally (see
    `config.py`), so a real dev setup would otherwise leak into every test
    run. Each entry here has bitten this exact way once already:

      * `allow_upload_host=True` made tests silently upload real image bytes
        over the network (14x slower suite, real third-party calls).
      * `face_only_verify_enabled`/`high_face_similarity_priority` set to a
        non-default value made tests assert against the wrong numbers.

    A test that specifically wants non-default behaviour opts in with its
    own `monkeypatch.setattr(settings, ...)` in the test body, which runs
    after this fixture and so still takes effect.
    """
    from facechain.config import settings
    monkeypatch.setattr(settings, "allow_upload_host", False)
    monkeypatch.setattr(settings, "local_image_base_url", "")
    monkeypatch.setattr(settings, "luxand_api_key", "")
    monkeypatch.setattr(settings, "high_face_similarity_priority", 0.75)
    monkeypatch.setattr(settings, "face_only_verify_enabled", False)
    monkeypatch.setattr(settings, "face_only_verify_threshold", 0.50)
