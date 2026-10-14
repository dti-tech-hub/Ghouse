"""Check that the README's factual claims match the repository.

A README that lies is worse than no README. Every number and file path it
quotes is verified here against the live system, the source tree and the
measurement data.

    python scripts/verify_readme.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

README = ROOT / "README.md"
DATA = ROOT / "docs" / "paper_data"


def check_claims() -> list[str]:
    problems: list[str] = []
    if not README.exists():
        return ["README.md is missing"]

    text = README.read_text(encoding="utf-8")

    # ---- 1. every referenced file path exists ------------------------------
    # Look inside each backtick span, because a reference is often a whole
    # command such as `python scripts/serve.py --status`.
    for span in re.findall(r"`([^`\n]+)`", text):
        for match in re.finditer(r"[A-Za-z0-9_.\-/]+\.(?:py|md|yaml|yml|txt|toml)", span):
            reference = match.group(0)
            if reference.startswith("http"):
                continue
            if (ROOT / reference).exists():
                continue
            # a bare filename is fine as long as it is unambiguous in the tree
            matches = [p for p in ROOT.rglob(Path(reference).name)
                       if "__pycache__" not in p.parts and ".venv" not in p.parts]
            if not matches:
                problems.append(f"README references a missing file: {reference}")
            elif len(matches) > 1:
                problems.append(
                    f"README references {reference!r} ambiguously "
                    f"({len(matches)} files share that name)"
                )

    # ---- 2. every referenced directory exists ------------------------------
    for match in re.finditer(r"`([A-Za-z0-9_.-]+/)`", text):
        candidate = ROOT / match.group(1)
        if not candidate.is_dir():
            problems.append(f"README references a missing directory: {match.group(1)}")

    # ---- 3. structural counts ----------------------------------------------
    def count(pattern: str) -> int:
        return sum(1 for _ in ROOT.glob(pattern) if "__pycache__" not in _.parts)

    pages = count("dashboard/pages/*.py")
    configs = count("experiments/configs/*.yaml")
    if f"| {pages} |" not in text and f"{pages} pages" not in text and f"{pages}-page" not in text:
        problems.append(f"README does not state the correct dashboard page count ({pages})")
    if f"{configs} configs" not in text and f"{configs} seeded" not in text:
        problems.append(f"README does not state the correct experiment config count ({configs})")

    # ---- 4. live system facts ----------------------------------------------
    try:
        from dashboard.api_client import ApiClient

        api = ApiClient("http://127.0.0.1:8012")
        health = api.health()
        if f"{health['documents']} documents" not in text:
            problems.append(f"README document count is wrong (live: {health['documents']})")
        if f"{health['chunks']} chunks" not in text:
            problems.append(f"README chunk count is wrong (live: {health['chunks']})")
    except Exception as exc:
        problems.append(f"backend not reachable, could not check corpus facts: {exc}")

    # ---- 5. measured numbers must match the data ---------------------------
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    overall = results["overall"]
    by_mode = json.loads((DATA / "by_mode.json").read_text(encoding="utf-8"))
    delta = by_mode["groundedness_delta_mode3"]
    taxonomy = json.loads((DATA / "taxonomy.json").read_text(encoding="utf-8"))
    detection = json.loads((DATA / "detection.json").read_text(encoding="utf-8"))
    faults = json.loads((DATA / "faults.json").read_text(encoding="utf-8"))

    expected = {
        "detection rate": f"{overall['detection_rate_pct']:g}",
        "top-1 accuracy": f"{overall['top1_accuracy_pct']:g}",
        "top-k accuracy": f"{overall['topk_accuracy_pct']:g}",
        "recovery success": f"{overall['recovery_success_pct_of_attempted']:g}",
        "abstention rate": f"{overall['abstention_rate_pct']:g}",
        "diagnosis p50": str(overall["diagnosis_latency_ms"]["p50"]),
        "added p50": str(overall["added_latency_ms"]["p50"]),
        "added p95": str(overall["added_latency_ms"]["p95"]),
        "e2e p50": str(overall["latency_ms"]["p50"]),
        "recovered count": str(overall["recovered"]),
        "attempted count": str(overall["runs_with_recovery_attempts"]),
        "sweep runs": str(overall["runs"]),
        "subtypes": str(taxonomy["subtype_count"]),
        "rules": str(detection["rule_id_count"]),
        "faults": str(len(faults)),
        "delta mean": f"{delta['mean_delta']:g}",
        "delta improved": str(delta["improved"]),
        "delta worsened": str(delta["worsened"]),
    }
    for label, value in expected.items():
        if value not in text:
            problems.append(f"README is missing or has a stale value for {label}: {value}")

    mode_runs = {row["mode"].split("_")[0]: row["runs"] for row in by_mode["by_mode"]}
    for short, count in mode_runs.items():
        if f"| `{short}`" in text and f"| {count} |" not in text:
            problems.append(f"README mode table has the wrong run count for {short} (live: {count})")

    # ---- 6. test count ------------------------------------------------------
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    collected = sum(
        int(m.group(1))
        for m in (re.match(r"tests/\S+\.py: (\d+)", line) for line in result.stdout.splitlines())
        if m
    )
    if collected and f"{collected} tests" not in text:
        problems.append(f"README test count is wrong (live: {collected})")

    # ---- 7. integrity: no fabricated-claim language ------------------------
    lowered = text.lower()
    for forbidden in ("state-of-the-art", "sota ", "production-ready", "outperforms"):
        if forbidden in lowered:
            problems.append(f"README contains an unsupported claim: {forbidden.strip()!r}")
    if "DEMO / MOCK RESULTS" not in text:
        problems.append("README does not disclose the mock-provider result label")

    return problems


def main() -> int:
    problems = check_claims()
    print("=" * 74)
    print(" README verification")
    print("=" * 74)
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}")
        print(f"\n{len(problems)} problem(s).")
        return 1
    print("  OK  every file path, count and measured value in README.md is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
