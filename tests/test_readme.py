"""The README must stay true to the repository.

`docs/CONFERENCE_PAPER.md` is checked against its data by tests/test_paper.py.
This file applies the same discipline to `README.md`: every file path it names
must exist, and every number it quotes must match the live system and the
measurement data. A README that lies is worse than no README.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DATA = ROOT / "docs" / "paper_data"


def readme_text() -> str:
    assert README.exists(), "README.md is missing"
    return README.read_text(encoding="utf-8")


def test_readme_exists_and_is_well_structured():
    text = readme_text()
    assert len(text.splitlines()) > 150, "the README lost its body"
    for heading in ("Why", "Quick start", "What it does", "Measured results",
                    "Project layout", "Design rules"):
        assert heading in text, f"missing README section: {heading}"


def test_readme_has_no_mojibake():
    """Regression: the previous README shipped with corrupted box-drawing chars."""
    text = readme_text()
    for bad in ("\ufffd", "â€", "Ã"):
        assert bad not in text, f"README contains corrupted characters: {bad!r}"


def test_readme_discloses_the_mock_result_label():
    assert "DEMO / MOCK RESULTS" in readme_text(), (
        "the README must state that reported results come from the mock provider"
    )


def test_readme_claims_no_api_key_is_required():
    text = readme_text()
    assert "No API key" in text or "no API key" in text
    assert "OPENAI_API_KEY" in text, "the README should show how to use a real provider"


def test_measured_numbers_match_the_data():
    text = readme_text()
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    overall = results["overall"]
    by_mode = json.loads((DATA / "by_mode.json").read_text(encoding="utf-8"))
    delta = by_mode["groundedness_delta_mode3"]

    expected = {
        "detection rate": f"{overall['detection_rate_pct']:g}",
        "top-1 accuracy": f"{overall['top1_accuracy_pct']:g}",
        "top-k accuracy": f"{overall['topk_accuracy_pct']:g}",
        "recovery success": f"{overall['recovery_success_pct_of_attempted']:g}",
        "abstention rate": f"{overall['abstention_rate_pct']:g}",
        "diagnosis p50": str(overall["diagnosis_latency_ms"]["p50"]),
        "added p50": str(overall["added_latency_ms"]["p50"]),
        "added p95": str(overall["added_latency_ms"]["p95"]),
        "recovered count": str(overall["recovered"]),
        "groundedness delta": f"{delta['mean_delta']:g}",
        "delta improved": str(delta["improved"]),
    }
    stale = [f"{label} = {value}" for label, value in expected.items() if value not in text]
    assert not stale, "the README quotes stale values: " + ", ".join(stale)


def test_structural_counts_match_the_tree():
    text = readme_text()
    pages = len([p for p in (ROOT / "dashboard" / "pages").glob("*.py")])
    configs = len(list((ROOT / "experiments" / "configs").glob("*.yaml")))
    faults = len(json.loads((DATA / "faults.json").read_text(encoding="utf-8")))
    subtypes = json.loads((DATA / "taxonomy.json").read_text(encoding="utf-8"))["subtype_count"]
    rules = json.loads((DATA / "detection.json").read_text(encoding="utf-8"))["rule_id_count"]

    for label, value in (
        ("dashboard pages", pages),
        ("experiment configs", configs),
        ("faults", faults),
        ("subtypes", subtypes),
        ("rules", rules),
    ):
        assert str(value) in text, f"the README does not state the current {label} count ({value})"


def test_verifier_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/verify_readme.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "verify_readme.py reported a mismatch:\n" + result.stdout
    )


@pytest.mark.parametrize(
    "forbidden",
    ["state-of-the-art", "sota ", "production-ready", "outperforms", "battle-tested"],
)
def test_readme_avoids_unsupported_claims(forbidden):
    assert forbidden not in readme_text().lower(), (
        f"the README contains an unsupported claim: {forbidden.strip()!r}"
    )
