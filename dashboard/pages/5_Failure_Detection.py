"""Failure detection overview: counts by stage, class and severity."""


import streamlit as st
import pandas as pd

from dashboard.common import client, guard, page


@guard
def main() -> None:
    page("Failure Detection", "🚨")
    api = client()
    metrics = api.metrics()

    cols = st.columns(6)
    cols[0].metric("Failures", metrics["counts"].get("failures", 0))
    cols[1].metric("Diagnoses", metrics["counts"].get("diagnoses", 0))
    cols[2].metric("Runs", metrics["counts"].get("runs", 0))
    cols[3].metric("Recoveries", metrics["counts"].get("recovery_attempts", 0))
    cols[4].metric("Successful", metrics["counts"].get("successful_recoveries", 0))
    cols[5].metric("Abstentions", metrics["counts"].get("abstentions", 0))

    if not metrics["failures_by_class"]:
        st.info("No failures recorded yet.")
        return

    by_class = pd.DataFrame(
        [{"failure_class": k, "count": v} for k, v in metrics["failures_by_class"].items()]
    ).sort_values("count", ascending=False)
    by_stage = pd.DataFrame(
        [{"stage": k, "count": v} for k, v in metrics["failures_by_stage"].items()]
    ).sort_values("count", ascending=False)
    by_severity = pd.DataFrame(
        [{"severity": k, "count": v} for k, v in metrics["severity_distribution"].items()]
    )

    left, right = st.columns(2)
    left.subheader("Failures by class")
    left.bar_chart(by_class.set_index("failure_class"))
    right.subheader("Failures by stage")
    right.bar_chart(by_stage.set_index("stage"))
    st.subheader("Severity distribution")
    st.bar_chart(by_severity.set_index("severity"))

    st.subheader("Recent diagnoses")
    diagnoses = api.diagnoses(limit=200)
    rows = [
        {
            "diagnosis_id": d["diagnosis_id"],
            "run_id": d["run_id"],
            "class": d["failure_class"],
            "subtype": d["subtype"],
            "stage": d["stage"],
            "severity": d["severity"],
            "confidence": d["confidence"],
            "tier": d["tier"],
        }
        for d in diagnoses
        if d.get("failure_class") not in (None, "NONE")
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


main()