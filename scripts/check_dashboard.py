"""Execute every dashboard page with Streamlit's AppTest and report errors.

This is the check that catches ``NameError: name 'st' is not defined`` and
any other runtime failure in a page, without needing a browser.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# the pages talk to the backend over HTTP; default to the local one
os.environ.setdefault("ARR_API_URL", os.environ.get("ARR_API_URL", "http://127.0.0.1:8012"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def run_page(path: Path) -> tuple[bool, str]:
    app = AppTest.from_file(str(path), default_timeout=120)
    try:
        app.run()
    except Exception as exc:  # pragma: no cover - surfaced in the report
        return False, f"{type(exc).__name__}: {exc}"
    if app.exception:
        first = app.exception[0]
        return False, f"{first.type}: {first.message}"
    # dashboard.common.guard renders failures with st.error() instead of raising,
    # so an unreachable backend would otherwise report "ok" for every page.
    if app.error:
        return False, f"rendered {len(app.error)} error box(es): {app.error[0].value}"
    return True, "ok"


def main() -> int:
    base_url = os.environ.get("ARR_API_URL", "http://127.0.0.1:8012").rstrip("/")
    print(f"backend: {base_url}")
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
            health = json.loads(response.read())
        print(
            f"  reachable: {health['provider']}/{health['model']} · "
            f"{health['documents']} documents / {health['chunks']} chunks · "
            f"{health.get('labels', {}).get('results', '')}"
        )
    except Exception as exc:
        print(f"  UNREACHABLE - {exc}")
        print("  start it first:  python scripts/serve.py")
        return 2

    pages = sorted((ROOT / "dashboard" / "pages").glob("*.py"))
    entry = ROOT / "dashboard" / "app.py"
    failures = 0
    for page in [entry, *pages]:
        ok, detail = run_page(page)
        status = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  [{status}] {page.relative_to(ROOT).as_posix():<42} {detail}")
    print()
    print(f"{len(pages) + 1} page(s) checked, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())