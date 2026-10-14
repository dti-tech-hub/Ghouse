"""Capture the live system inventory used by the conference-paper document.

Every value written here comes from the running system: the FastAPI backend,
the SQLite store, or the source tree. Nothing is hard-coded, so the document
can be regenerated after any change and will stay honest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "paper_data"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, payload) -> Path:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    size = path.stat().st_size
    print(f"  wrote {path.relative_to(ROOT).as_posix():<42} {size:>8,} bytes")
    return path


def main() -> int:
    api_base = os.environ.get("ARR_API_URL", "http://127.0.0.1:8012").rstrip("/")
    from dashboard.api_client import ApiClient

    api = ApiClient(api_base)
    print(f"backend: {api_base}")

    # ---- system facts -----------------------------------------------------
    health = api.health()
    config = api.get("/config")
    dump("system", {"health": health, "config": config})

    # ---- fault catalogue --------------------------------------------------
    dump("faults", api.faults())

    # ---- source-tree size metrics ----------------------------------------
    counts: dict[str, int] = {}
    for pattern, key in (
        ("agentic_rag_recovery/**/*.py", "core"),
        ("backend/*.py", "backend"),
        ("dashboard/**/*.py", "dashboard"),
        ("scripts/*.py", "scripts"),
        ("tests/*.py", "tests"),
        ("experiments/configs/*.yaml", "experiment_configs"),
    ):
        files = [p for p in ROOT.glob(pattern) if "__pycache__" not in p.parts]
        counts[key] = len(files)
        counts[f"{key}_loc"] = sum(
            len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in files
        )
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        import re as _re

        counts["collected_tests"] = sum(
            int(m.group(1))
            for m in (_re.match(r"tests/\S+\.py: (\d+)", line) for line in out.stdout.splitlines())
            if m
        )
    except Exception:
        counts["collected_tests"] = 0
    dump("codebase", counts)

    # ---- taxonomy / strategy registries ----------------------------------
    from agentic_rag_recovery.taxonomy import FailureTaxonomy, class_for_subtype
    from agentic_rag_recovery.recovery.strategies import STRATEGIES

    definitions = FailureTaxonomy.all()
    by_class: dict[str, list[str]] = {}
    for subtype, definition in definitions.items():
        by_class.setdefault(class_for_subtype(subtype).value, []).append(subtype)
    dump("taxonomy", {
        "stages": FailureTaxonomy.stages(),
        "subtype_count": len(definitions),
        "count_by_failure_class": {k: len(v) for k, v in sorted(by_class.items())},
        "subtypes_by_class": {k: sorted(v) for k, v in sorted(by_class.items())},
        "subtypes": {
            subtype: {
                "failure_class": class_for_subtype(subtype).value,
                "stage": d.stage.value,
                "default_severity": d.default_severity.value,
                "recovery_actions": [a.value for a in d.recovery_actions],
                "evidence_hints": list(d.evidence_hints),
                "cost_hint": d.cost_hint,
                "reusable_stage": d.reusable_stage.value if d.reusable_stage else None,
            }
            for subtype, d in sorted(definitions.items())
        },
    })
    dump("strategies", {
        "count": len(STRATEGIES),
        "actions": sorted(a.value if hasattr(a, "value") else str(a) for a in STRATEGIES),
    })

    # ---- configured thresholds -------------------------------------------
    import yaml

    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dump("config_yaml", cfg)

    print("inventory captured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
