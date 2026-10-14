"""Diagnosis page: evidence, confidence, tier and schema validation."""


import streamlit as st
from dashboard.common import client, guard, page


@guard
def main() -> None:
    page("Diagnosis", "🧠")
    api = client()
    diagnoses = api.diagnoses(limit=300)
    failures = [d for d in diagnoses if d.get("failure_class") not in (None, "NONE")]

    if not failures:
        st.info("No diagnoses recorded yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Diagnoses", len(failures))
    cols[1].metric("Rule tier", sum(1 for d in failures if d["tier"] == "rule"))
    cols[2].metric("LLM tier", sum(1 for d in failures if d["tier"] == "llm"))
    cols[3].metric("Fallback", sum(1 for d in failures if d["tier"] == "fallback"))

    rows = [
        {
            "diagnosis_id": d["diagnosis_id"],
            "class": d["failure_class"],
            "subtype": d["subtype"],
            "stage": d["stage"],
            "severity": d["severity"],
            "confidence": d["confidence"],
            "tier": d["tier"],
        }
        for d in failures
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect diagnosis", [d["diagnosis_id"] for d in failures])
    detail = api.diagnosis(selected)
    st.json(detail, expanded=True)
    st.markdown("**Evidence**")
    for item in detail.get("evidence", []):
        st.markdown(f"- {item}")
    st.markdown("**Raw LLM output**")
    raw = detail.get("raw_llm_output")
    st.code(raw if raw else "(rule-based diagnosis, no LLM call)", language="json")


main()