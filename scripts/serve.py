"""Start the backend API and the Streamlit dashboard on free ports.

Usage:
    python scripts/serve.py                # start both, print the URLs
    python scripts/serve.py --status       # show what is already running
    python scripts/serve.py --stop         # stop both

The dashboard is a thin HTTP client, so both processes must be running.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "storage" / "logs"
PIDFILE = ROOT / "storage" / "logs" / "serve.pids"


def free_port(preferred: int = 8010) -> int:
    """Return a TCP port that is actually bindable right now."""
    for candidate in (preferred, 8012, 8020, 8030, 8100, 8200, 8501, 8600, 9000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def health(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
            return json.loads(response.read())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"error": str(exc)}


def read_pids() -> dict:
    if not PIDFILE.exists():
        return {}
    try:
        return json.loads(PIDFILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_pids(data: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args(argv)

    pids = read_pids()

    if args.stop:
        for name in ("api", "ui"):
            pid = pids.get(name)
            if pid is None:
                continue
            try:
                os.kill(int(pid), 15)
                print(f"stopped {name} (pid {pid})")
            except Exception as exc:
                print(f"could not stop {name} (pid {pid}): {exc}")
        if PIDFILE.exists():
            PIDFILE.unlink()
        return 0

    if args.status:
        api_port = pids.get("api_port")
        ui_port = pids.get("ui_port")
        for name in ("api", "ui"):
            pid = pids.get(name)
            if pid is None:
                print(f"{name:<4} not started")
                continue
            print(f"{name:<4} pid={pid} alive={_pid_alive(int(pid))}")
        if api_port:
            print(f"     api      http://127.0.0.1:{api_port}  {health(f'http://127.0.0.1:{api_port}')}")
        if ui_port:
            print(f"     ui       http://127.0.0.1:{ui_port}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    existing = read_pids()
    if existing.get("api") and existing.get("ui") and _pid_alive(int(existing["api"])) and _pid_alive(int(existing["ui"])):
        api_base = f"http://127.0.0.1:{existing['api_port']}"
        ui_base = f"http://127.0.0.1:{existing['ui_port']}"
        print("already running - nothing to start")
        print(f"  dashboard : {ui_base}")
        print(f"  API docs  : {api_base}/docs")
        return 0

    api_port = free_port(args.api_port)
    ui_port = args.ui_port if not _in_use(args.ui_port) else free_port(8502)
    api_base = f"http://127.0.0.1:{api_port}"
    ui_base = f"http://127.0.0.1:{ui_port}"

    print(f"starting backend API on {api_base} ...")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=str(ROOT),
        stdout=(LOGS / "api.out.log").open("w", encoding="utf-8"),
        stderr=(LOGS / "api.err.log").open("w", encoding="utf-8"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if not wait_for(f"{api_base}/health"):
        print("backend failed to start; last log lines:")
        print((LOGS / "api.err.log").read_text(encoding="utf-8")[-2000:])
        return 1
    info = health(api_base)
    print(f"  backend up: {info.get('provider')}/{info.get('model')} "
          f"docs={info.get('documents')} chunks={info.get('chunks')} label={info.get('labels', {}).get('results')}")

    environment = dict(os.environ)
    environment["ARR_API_URL"] = api_base
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    print(f"starting dashboard on {ui_base} (ARR_API_URL={api_base}) ...")
    ui = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(ui_port), "--server.headless", "true",
         "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
        env=environment,
        stdout=(LOGS / "ui.out.log").open("w", encoding="utf-8"),
        stderr=(LOGS / "ui.err.log").open("w", encoding="utf-8"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if not wait_for(f"{ui_base}/_stcore/health"):
        print("dashboard failed to start; last log lines:")
        print((LOGS / "ui.err.log").read_text(encoding="utf-8")[-2000:])
        return 1
    print("  dashboard up")

    write_pids({"api": api.pid, "ui": ui.pid, "api_port": api_port, "ui_port": ui_port})
    print()
    print(f"  dashboard : {ui_base}")
    print(f"  API docs  : {api_base}/docs")
    print(f"  API health: {api_base}/health")
    print()
    print(f"  stop with : python scripts/serve.py --stop")
    print(f"  logs      : {LOGS}")
    return 0


def _in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _pid_alive(pid: int) -> bool:
    import subprocess as sp

    if os.name == "nt":
        result = sp.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in result.stdout
    return Path(f"/proc/{pid}").exists()


if __name__ == "__main__":
    raise SystemExit(main())