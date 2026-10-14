"""The paper document must stay in step with the data it quotes.

`docs/CONFERENCE_PAPER.md` is written from `docs/paper_data/*.json`. If a run is
re-executed and the numbers move, the document is wrong until it is rewritten -
so this test fails loudly instead of letting a stale figure ship.

Regenerate everything with:

    python scripts/paper_fault_sweep.py 24
    python scripts/paper_aggregate.py
    python scripts/paper_by_mode.py
    python scripts/paper_experiments.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "paper_data"
DOC = ROOT / "docs" / "CONFERENCE_PAPER.md"

REQUIRED_DATA = [
    "results.json",
    "results.md",
    "by_mode.md",
    "fault_sweep.json",
    "by_mode.json",
    "experiments.json",
    "taxonomy.json",
    "detection.json",
    "codebase.json",
    "config_yaml.json",
    "system.json",
    "faults.json",
]


def test_paper_data_files_exist():
    missing = [name for name in REQUIRED_DATA if not (DATA / name).exists()]
    assert not missing, (
        f"missing measurement files: {missing}. Regenerate with the "
        "scripts/paper_*.py commands listed in docs/CONFERENCE_PAPER.md"
    )


def test_document_exists_and_is_substantial():
    assert DOC.exists()
    text = DOC.read_text(encoding="utf-8")
    assert len(text.splitlines()) > 200, "the document lost its body"
    for section in ("Abstract", "System Design", "Experimental Setup", "Results",
                    "Discussion", "Reproducibility", "Conclusion"):
        assert section in text, f"missing section: {section}"


@pytest.mark.parametrize("table", ["results.md", "by_mode.md"])
def test_every_generated_row_appears_in_the_document(table):
    """The results tables are generated; the document must reproduce them verbatim."""
    text = DOC.read_text(encoding="utf-8")
    rows = [
        line.strip()
        for line in (DATA / table).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    missing = [row for row in rows if row not in text]
    assert not missing, (
        f"{len(missing)} generated row(s) from {table} are absent from the document, e.g.\n  "
        + "\n  ".join(missing[:5])
    )


def test_artifact_inventory_matches_the_store():
    """The inventory table quotes live store totals; a reset database invalidates it."""
    experiments = json.loads((DATA / "experiments.json").read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    totals = experiments["totals"]
    inventory = {
        "recorded runs in the evaluation store": totals["runs"],
        "recorded diagnoses": totals["diagnoses"],
        "recorded recovery attempts": totals["recovery_attempts"],
        "recorded fault injections": totals["fault_injections"],
        "recorded metric rows": totals["metrics_rows"],
        "recorded trace spans": totals["trace_spans"],
        "experiment configurations": len(experiments["experiments"]),
    }
    stale = [
        f"{label} = {value:,}"
        for label, value in inventory.items()
        if f"| {label} | {value:,}".replace(",", " ") not in text
    ]
    assert not stale, (
        "the artifact inventory is stale: " + ", ".join(stale)
        + ". Re-run scripts/run_experiments.py --all and update the table."
    )


def test_verifier_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/verify_paper_numbers.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "verify_paper_numbers.py reported a mismatch:\n" + result.stdout
    )


def test_headline_figures_match_the_data():
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    overall = results["overall"]
    text = DOC.read_text(encoding="utf-8")
    for key in ("detection_rate_pct", "top1_accuracy_pct", "topk_accuracy_pct",
                "abstention_rate_pct", "recovery_success_pct_of_attempted"):
        value = overall[key]
        assert f"{value:g}" in text, f"{key}={value} is not quoted in the document"


def test_results_carry_the_mock_label():
    """The paper must not present mock-provider results as real-model results."""
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    assert "DEMO" in str(results["label"]).upper()
    text = DOC.read_text(encoding="utf-8")
    assert "DEMO / MOCK RESULTS" in text, "the document does not state the result label"


def test_document_does_not_claim_unrun_results():
    text = DOC.read_text(encoding="utf-8").lower()
    for forbidden in ("sota", "state-of-the-art", "outperforms gpt", "production scale"):
        assert forbidden not in text, (
            f"the document contains an unsupported claim: {forbidden!r}"
        )


@pytest.mark.parametrize("name", ["results.json", "by_mode.json", "experiments.json"])
def test_measurements_are_non_trivial(name):
    payload = json.loads((DATA / name).read_text(encoding="utf-8"))
    text = json.dumps(payload)
    assert len(text) > 500, f"{name} looks empty; the measurements did not run"
