from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19"


def test_v16_rc5_matches_the_notebook_artifact() -> None:
    baseline = Path(__file__).parents[1] / "baselines" / "v16_rc5" / "main.py"
    payload = baseline.read_bytes()

    assert len(payload) == 18_946
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
    compile(payload, str(baseline), "exec")
