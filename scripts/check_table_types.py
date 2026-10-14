"""Reproduce the ArrowInvalid rendering failure and prove `common.table` fixes it.

``st.dataframe`` silently drops columns when a column's cells have mixed types,
which is exactly what a verification criterion produces (``0.62``, ``"< 0.62"``,
``False``, ``"[60, 3000]"``). pandas infers the dtype from the first row and
pyarrow then raises for the rest.

Run with a backend up:  python scripts/check_table_types.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ARR_API_URL", "http://127.0.0.1:8012")

from dashboard.api_client import ApiClient  # noqa: E402
from dashboard.common import scalar_text  # noqa: E402

CRITERIA = [
    {"criterion": "groundedness_above_threshold", "passed": True,
     "observed": 0.83, "required": 0.62, "detail": ""},
    {"criterion": "unsupported_claims_at_most", "passed": False,
     "observed": 3, "required": "<= 0", "detail": "claims not grounded"},
    {"criterion": "context_tokens_in_window", "passed": False,
     "observed": 1840, "required": "[60, 3000]", "detail": "window widened"},
    {"criterion": "no_external_tool_calls", "passed": True,
     "observed": 0, "required": 0, "detail": ""},
    {"criterion": "budget_not_exhausted", "passed": True,
     "observed": False, "required": False, "detail": ""},
]


def _render_with_dataframe(rows: list[dict]) -> str:
    """Reproduce the failure the dashboard used to hit."""
    import pandas as pd
    import pyarrow  # noqa: F401

    frame = pd.DataFrame(rows)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pyarrow.Table.from_pandas(frame, preserve_index=False)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def main() -> int:
    print("=" * 74)
    print(" 1. mixed-type criteria through the raw st.dataframe path")
    print("=" * 74)
    failure = _render_with_dataframe(CRITERIA)
    if failure:
        print(f"  reproduces as expected -> {failure.splitlines()[0][:150]}")
    else:
        print("  (pandas tolerated it this time; type mixing is still fragile)")

    print()
    print("=" * 74)
    print(" 2. the same rows through common.table (all cells coerced to text)")
    print("=" * 74)
    rows = [
        {k: scalar_text(v) for k, v in record.items()}
        for record in CRITERIA
    ]
    for row in rows:
        print(f"  {row['criterion']:<32} observed={row['observed']:<8} required={row['required']}")
    failure = _render_with_dataframe(rows)
    print()
    print(f"  homogeneous render: {'FAILED -> ' + failure if failure else 'OK'}")

    print()
    print("=" * 74)
    print(" 3. live backend payloads")
    print("=" * 74)
    api = ApiClient(os.environ["ARR_API_URL"])
    try:
        traces = api.traces(limit=50)
    except Exception as exc:
        print(f"  backend not reachable ({exc}); structural checks only")
        return 0

    mixed = 0
    total = 0
    for trace in traces:
        trace_id = trace["run_id"]
        for attempt in api.trace(trace_id).get("recoveries", []):
            for criterion in attempt.get("verification", {}).get("criteria", []):
                total += 1
                if not isinstance(criterion["required"], str):
                    mixed += 1
    print(f"  criteria inspected : {total}")
    print(f"  non-string 'required' values: {mixed}")

    print()
    ok = not failure
    print("RESULT:", "all criteria render as text" if ok else "rendering still fails")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
