from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tri_provider_acceptance import API_KEY_ENV, LIVE_ENV, run_acceptance


def test_tri_provider_live_acceptance_harness() -> None:
    if os.environ.get(LIVE_ENV) != "1" or not os.environ.get(API_KEY_ENV):
        pytest.skip(
            f"live acceptance disabled; set {LIVE_ENV}=1 and {API_KEY_ENV} to run"
        )

    summary = run_acceptance()

    assert summary["success"] is True, summary.get("artifact_dir")
