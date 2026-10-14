#!/usr/bin/env python
"""Run one or more experiment configurations and print the results table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import ensure_demo_ready, create_application  # noqa: E402
from agentic_rag_recovery.evaluation.runner import ExperimentRunner  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="configuration name (without .yaml)")
    parser.add_argument("--all", action="store_true", help="run every configuration")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default="", help="write the full results to this file")
    args = parser.parse_args(argv)

    app = create_application()
    ensure_demo_ready(app)
    runner = ExperimentRunner(app)

    names = runner.available_configs() if args.all else [args.config or "baseline"]
    if not args.all and not args.config:
        names = ["baseline"]

    results = []
    header = (
        f"{'experiment':<26}{'runs':>6}{'diag acc':>10}{'rec succ':>10}"
        f"{'grounded':>10}{'answerF1':>10}{'abstain':>9}{'+ms':>8}{'cost/rec':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        try:
            config = runner.load_config(name)
        except FileNotFoundError as exc:
            print(f"{name:<26} CONFIG NOT FOUND: {exc}")
            continue
        if args.seed is not None:
            config["seed"] = args.seed
        if args.repetitions is not None:
            config["repetitions"] = args.repetitions
        if args.limit is not None:
            config["limit"] = args.limit
        result = runner.run(config)
        m3 = result.mode3
        print(
            f"{name:<26}{m3.total_runs:>6}{m3.diagnosis_accuracy:>10.3f}"
            f"{m3.recovery_success_rate:>10.3f}{m3.groundedness:>10.3f}"
            f"{m3.answer_correctness:>10.3f}{m3.abstention_rate:>9.3f}"
            f"{m3.added_latency_ms:>8.1f}{m3.cost_per_recovery:>11.6f}"
        )
        results.append(result.to_dict())

    print(f"\nlabel: {app.settings.labels.get('results')}")
    print(f"available configs: {', '.join(runner.available_configs())}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())