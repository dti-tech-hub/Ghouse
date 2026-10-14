"""Metrics dashboard: the six headline metrics plus supporting detail."""


import streamlit as st
import pandas as pd

from dashboard.common import client, guard, label_banner, page

HEADLINE = [
    ("diagnosis_accuracy", "Diagnosis accuracy", "{:.3f}"),
    ("recovery_success_rate", "Recovery success rate", "{:.3f}"),
    ("groundedness", "Groundedness", "{:.3f}"),
    ("added_latency_ms", "Added latency (ms)", "{:.1f}"),
    ("cost_per_recovery", "Cost per recovery", "{:.6f}"),
    ("abstention_rate", "Abstention rate", "{:.3f}"),
]


@guard
def main() -> None:
    page("Metrics", "📈")
    label_banner()
    api = client()
    metrics = api.metrics()
    aggregates = metrics.get("aggregates", {})

    if not aggregates:
        st.info("No metrics recorded yet. Run a query or an experiment first.")
        return

    cols = st.columns(len(HEADLINE))
    for column, (key, title, fmt) in zip(cols, HEADLINE):
        if key in aggregates:
            column.metric(title, fmt.format(aggregates[key]))

    st.subheader("All recorded aggregates")
    st.dataframe(
        [{"metric": k, "value": v} for k, v in sorted(aggregates.items())],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Per-run metrics")
    rows = api.per_run_metrics(limit=200)
    frame = pd.DataFrame(
        [{"metric": r["name"], "value": r["value"], "run_id": r["run_id"], "group": r["group_name"]} for r in rows]
    )
    if not frame.empty:
        pivot = frame.pivot_table(index="metric", values="value", aggfunc="mean")
        st.dataframe(pivot, use_container_width=True)
        wanted = [key for key, _, _ in HEADLINE if key in pivot.index]
        if wanted:
            st.bar_chart(pivot.loc[wanted])

    st.subheader("Result integrity")
    st.info(
        "Numbers on this page are computed from executed runs stored in SQLite. "
        f"Figures produced with the deterministic mock provider carry the label "
        f"**{metrics.get('label')}** and are not comparable with real-LLM results."
    )


main()