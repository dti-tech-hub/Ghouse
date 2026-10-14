#!/usr/bin/env python
"""Full project validation: demo + experiments + tests + service smoke checks.

This is the script to run before a demonstration. It executes real work and
reports honestly; it never fabricates a result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def step(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def run(command: list[str], label: str) -> int:
    print(f"\n$ {' '.join(command)}")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(ROOT))
    elapsed = time.perf_counter() - started
    print(f"-> {label}: exit {result.returncode} in {elapsed:.1f}s")
    return result.returncode


def _looks_like_this_backend(base: str) -> bool:
    """True only if ``base`` serves *our* API.

    Port 8000 is frequently occupied by an unrelated service on developer
    machines, so a 200 on /health is not sufficient evidence.
    """
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=3) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read())
    except Exception:
        return False
    return "provider" in payload and "chunks" in payload


def _find_backend() -> str | None:
    """Return a usable backend base URL, or None."""
    configured = os.environ.get("ARR_API_URL")
    candidates = [configured] if configured else [
        "http://127.0.0.1:8012",
        "http://127.0.0.1:8000",
    ]
    for candidate in candidates:
        if candidate and _looks_like_this_backend(candidate.rstrip("/")):
            return candidate.rstrip("/")
    return None


def _wait_for(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="skip the Streamlit page checks (needs a running backend)",
    )
    args = parser.parse_args(argv)

    failures = 0
    python = sys.executable

    step("1/8  dependencies")
    for module in ("fastapi", "uvicorn", "streamlit", "numpy", "pandas", "pydantic", "yaml", "httpx", "pytest"):
        try:
            __import__(module)
            print(f"  OK   {module}")
        except ImportError:
            failures += 1
            print(f"  FAIL {module} - pip install -r requirements.txt")

    step("2/8  database")
    failures += run([python, "scripts/init_db.py", "--reset"], "init_db")

    step("3/8  ingestion + indexes")
    failures += run([python, "scripts/ingest.py", "--force"], "ingest")

    step("4/8  health")
    from agentic_rag_recovery.bootstrap import create_application, ensure_demo_ready

    app = create_application()
    status = ensure_demo_ready(app)
    health = app.orchestrator.health()
    print(f"  documents : {status['documents']}")
    print(f"  chunks    : {status['chunks']}")
    print(f"  provider  : {health['settings']['provider']}")
    print(f"  retrieval : {health['settings']['retrieval_method']} top_k={health['settings']['top_k']}")
    print(f"  labels    : {health['labels']}")

    step("5/8  end-to-end demonstration")
    failures += run([python, "run_demo.py", "--quick", "--no-colour"], "demo")

    if not args.skip_experiments:
        step("6/8  experiments")
        failures += run(
            [python, "scripts/run_experiments.py", "--config", "full_recovery", "--limit", "3"],
            "experiments",
        )

    if not args.skip_tests:
        step("7/8  test suite")
        failures += run([python, "-m", "pytest", "-q"], "pytest")

    step("8/8  dashboard pages")
    if args.skip_dashboard:
        print("  skipped (--skip-dashboard)")
    else:
        started_here = None
        base_url = _find_backend()
        if base_url:
            print(f"  using the running backend at {base_url}")
        else:
            try:
                __import__("streamlit")
            except ImportError:
                failures += 1
                print("  FAIL streamlit is not installed - pip install -r requirements.txt")
            else:
                print("  no backend found - starting a temporary one on :8099")
                started_here = subprocess.Popen(
                    [python, "-m", "uvicorn", "backend.main:app", "--port", "8099",
                     "--log-level", "warning"],
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _wait_for("http://127.0.0.1:8099/health", timeout=90)
                base_url = "http://127.0.0.1:8099"
        env_api = os.environ.get("ARR_API_URL")
        if base_url:
            os.environ["ARR_API_URL"] = base_url
            failures += run([python, "scripts/check_dashboard.py"], "dashboard pages")
        if started_here is not None:
            started_here.terminate()
            try:
                started_here.wait(timeout=15)
            except Exception:
                started_here.kill()
            print("  temporary backend stopped")
        if env_api is None:
            os.environ.pop("ARR_API_URL", None)
        else:
            os.environ["ARR_API_URL"] = env_api

    step("RESULT")
    if failures:
        print(f"  {failures} step(s) reported a non-zero exit code - see the output above.")
    else:
        print("  All validation steps completed successfully.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())