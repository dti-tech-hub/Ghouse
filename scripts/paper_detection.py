"""Count the detection rules, signals and metrics defined in the core package.

These are static structural facts used in the paper's system description.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.detection import rules as rules_module  # noqa: E402
from agentic_rag_recovery.signals import collector as collector_module  # noqa: E402

DATA = ROOT / "docs" / "paper_data"


def main() -> int:
    import re

    source = (ROOT / "agentic_rag_recovery" / "detection" / "rules.py").read_text(encoding="utf-8")
    rule_ids = sorted(set(re.findall(r"R_[A-Z]+_\d{3}", source)))
    functions = [
        name for name in dir(rules_module)
        if name.startswith("rule_") and callable(getattr(rules_module, name))
    ]

    payload = {
        "rule_module": "agentic_rag_recovery/detection/rules.py",
        "rule_ids": rule_ids,
        "rule_id_count": len(rule_ids),
        "rule_functions": sorted(functions),
        "rule_function_count": len(functions),
        "collector_functions": sorted(
            name for name in dir(collector_module)
            if name.startswith("collect_") and callable(getattr(collector_module, name))
        ),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "detection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"rule ids        : {len(rule_ids)}")
    print(f"  by family      : " + ", ".join(
        f"{family}={sum(1 for r in rule_ids if r.startswith(f'R_{family}_'))}"
        for family in sorted({r.split('_')[1] for r in rule_ids})
    ))
    print(f"rule functions  : {len(functions)}")
    print(f"signal collectors: {len(payload['collector_functions'])}")
    print(f"wrote           : {(DATA / 'detection.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
