"""Print a summary of the SQLite database (verification helper)."""

import sqlite3

db = sqlite3.connect("storage/agentic_rag.db")
db.row_factory = sqlite3.Row

tables = db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
print("TABLES:", tables)
for table in (
    "documents",
    "chunks",
    "queries",
    "retrieval_runs",
    "retrieved_chunks",
    "agent_runs",
    "agent_steps",
    "trace_spans",
    "diagnoses",
    "recovery_attempts",
    "fault_injections",
    "abstentions",
    "experiments",
    "metrics",
    "models",
    "configurations",
):
    count = db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
    print("  %-20s %5d rows" % (table, count))

print()
print("DIAGNOSIS BY TIER:")
for row in db.execute("SELECT tier, COUNT(*) n FROM diagnoses GROUP BY tier"):
    print("  ", dict(row))
print("RECOVERY BY STATUS:")
for row in db.execute("SELECT status, COUNT(*) n FROM recovery_attempts GROUP BY status"):
    print("  ", dict(row))
print("FAULTS INJECTED:")
for row in db.execute("SELECT fault_type, COUNT(*) n FROM fault_injections GROUP BY fault_type ORDER BY fault_type"):
    print("  ", dict(row))
print("METRIC LABELS:")
for row in db.execute("SELECT DISTINCT label FROM metrics"):
    print("  ", dict(row))
print("RUN STATUSES:")
for row in db.execute("SELECT status, COUNT(*) n FROM agent_runs GROUP BY status"):
    print("  ", dict(row))