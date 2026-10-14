"""Trace viewer: stage-by-stage inspection of any recorded run."""


import streamlit as st
import pandas as pd

from dashboard.common import client, guard, page, table


@guard
def main() -> None:
    page("Agent Trace", "🔍")
    api = client()
    traces = api.traces(limit=100)
    if not traces:
        st.warning("No runs recorded yet. Use the Query Playground first.")
        return

    st.dataframe(
        [
            {
                "run_id": t["run_id"],
                "query": t["query"][:60],
                "mode": t["mode"],
                "status": t["status"],
                "abstained": t["abstained"],
                "total_ms": round(t.get("total_latency_ms", 0.0), 2),
                "added_ms": round(t.get("added_latency_ms", 0.0), 2),
                "created_at": t["created_at"],
            }
            for t in traces
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox("Inspect run", [t["run_id"] for t in traces])
    trace = api.trace(selected)
    st.session_state["last_trace"] = trace

    cols = st.columns(5)
    cols[0].metric("status", trace["status"])
    cols[1].metric("mode", trace["mode"])
    cols[2].metric("total ms", round(trace["total_latency_ms"], 2))
    cols[3].metric("added ms", round(trace["added_latency_ms"], 2))
    cols[4].metric("abstained", trace["abstained"])

    st.subheader("Steps")
    table(trace["steps"])

    st.subheader("Latency by stage")
    by_stage: dict = {}
    for step in trace["steps"]:
        by_stage.setdefault(step["stage"], 0.0)
        by_stage[step["stage"]] += step.get("latency_ms", 0.0)
    if by_stage:
        st.bar_chart(
            pd.DataFrame({"stage": list(by_stage), "latency_ms": list(by_stage.values())}).set_index("stage")
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Signals", "Diagnoses", "Recoveries", "Retrieved"])
    with tab1:
        st.json(trace["signals"], expanded=True)
    with tab2:
        st.json(trace["diagnoses"], expanded=False)
    with tab3:
        st.json(trace["recoveries"], expanded=False)
    with tab4:
        st.dataframe(trace["retrieved"], use_container_width=True, hide_index=True)
    if trace.get("spans"):
        st.subheader("Spans")
        table(trace["spans"])


main()