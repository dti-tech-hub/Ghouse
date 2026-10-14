"""Smoke-test the running services through their HTTP interfaces."""

import json
import sys
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8012"
UI = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8501"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode(errors="replace")


def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


print("=" * 74)
print(" SERVICES")
print("=" * 74)
print("  API health   :", get(f"{API}/health").strip()[:120])
print("  Dashboard    :", get(f"{UI}/_stcore/health").strip())
print("  Dashboard UI :", "served" if "Streamlit" in get(f"{UI}/") else "check logs")

print()
print("=" * 74)
print(" SCENARIO 1 - healthy agentic RAG (no fault)")
print("=" * 74)
d = post(
    f"{API}/query",
    {"query": "Within how many minutes must a Severity 1 incident be acknowledged by the on-call engineer?"},
)
print("  status      :", d["status"])
print("  answer      :", d["answer"][:170])
print("  retrieved   :", len(d["retrieved"]), "chunks | top coverage", d["signals"]["retrieval"]["top_overlap"])
print("  groundedness:", d["signals"]["generation"]["groundedness"],
      "| citation coverage", d["signals"]["generation"]["citation_coverage"])
print("  diagnosis   :", d["diagnoses"][0]["failure_class"], "| tier", d["diagnoses"][0]["tier"])
print("  recoveries  :", len(d["recoveries"]), "| abstained:", d["abstained"])

print()
print("=" * 74)
print(" SCENARIO 2 - GENERATION_CONSTRAINT_REMOVAL -> CONSTRAINED_REGENERATION")
print("=" * 74)
d = post(
    f"{API}/query",
    {
        "query": "What is the minimum password length and how many previous passwords must not be reused?",
        "faults": ["GENERATION_CONSTRAINT_REMOVAL"],
    },
)
print("  expected    : GENERATION_FAILURE/hallucination")
print(
    "  diagnosed   :",
    d["diagnoses"][0]["failure_class"] + "/" + str(d["diagnoses"][0]["subtype"]),
    "| tier", d["diagnoses"][0]["tier"],
    "| conf", d["diagnoses"][0]["confidence"],
    "| rule", d["diagnoses"][0]["rule_id"],
)
print("  evidence    :", " ; ".join(d["diagnoses"][0]["evidence"]))
for a in d["recoveries"]:
    v = a["verification"]
    print(
        "  recovery    :",
        a["action"],
        "->",
        a["status"],
        "| verified =", v.get("passed"),
        "| reused", len(a["skipped_stages"]), "stages",
        "| +", round(a["added_latency_ms"], 2), "ms",
    )
    print("                reused:", ", ".join(s.replace("_", " ") for s in a["skipped_stages"]) or "-")
print("  recovered   :", d["recovered"], "| abstained:", d["abstained"])
print("  groundedness:", d["signals"]["generation"]["groundedness"], "(after recovery)")
print("  answer      :", d["answer"][:170])

print()
print("=" * 74)
print(" SCENARIO 3 - unverifiable conflict -> bounded loop -> ABSTENTION")
print("=" * 74)
d = post(
    f"{API}/query",
    {
        "query": "When must a written postmortem be completed for a severity incident?",
        "faults": ["CONFLICTING_EVIDENCE"],
    },
)
print("  diagnosed   :", d["diagnoses"][0]["failure_class"] + "/" + str(d["diagnoses"][0]["subtype"]))
for a in d["recoveries"]:
    failed = [c["criterion"] for c in a["verification"].get("criteria", []) if not c["passed"]]
    print("  recovery    :", a["action"], "->", a["status"], "| failed:", failed or "-")
print("  abstained   :", d["abstained"])
if d.get("abstention"):
    ab = d["abstention"]
    print("  reason      :", ab["reason"])
    print("  attempted   :", [a for a in ab["attempted_recoveries"]],
          "|", ab["attempts_used"], "/", ab["max_attempts"])
    print("  message     :", ab["message"])
print("  answer      :", d["answer"][:150])

print()
print("=" * 74)
print(" OBSERVABILITY")
print("=" * 74)
m = json.loads(get(f"{API}/metrics"))
print("  counts      :", m["counts"])
print("  labels      :", m["label"])
tr = json.loads(get(f"{API}/traces?limit=3"))
print("  recent runs :", [(t["run_id"][:16], t["status"], round(t["total_latency_ms"], 1)) for t in tr["traces"]])
f = json.loads(get(f"{API}/faults"))
print("  fault catalogue:", len(f), "declared faults")

print()
print("All service checks completed.")