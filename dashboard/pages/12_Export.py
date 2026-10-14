"""Export: download traces, metrics and experiment results as JSON/CSV."""


import streamlit as st
import json

import pandas as pd

from dashboard.common import client, guard, page, table


@guard
def main() -> None:
    page("Export", "📤")
    api = client()

    st.subheader("Export a single run")
    traces = api.traces(limit=200)
    if traces:
        run_id = st.selectbox("Run", [t["run_id"] for t in traces])
        trace = api.trace(run_id)
        st.download_button(
            "Download trace JSON",
            data=json.dumps(trace, indent=2, default=str),
            file_name=f"{run_id}.json",
            mime="application/json",
        )
        table(trace["steps"])

    st.subheader("Export metrics")
    rows = api.per_run_metrics(limit=1000)
    frame = pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "metric": r["name"],
                "value": r["value"],
                "group": r["group_name"],
                "label": r["label"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    )
    if not frame.empty:
        st.download_button(
            "Download metrics CSV",
            data=frame.to_csv(index=False),
            file_name="metrics.csv",
            mime="text/csv",
        )
        st.dataframe(frame.head(200), use_container_width=True)

    st.subheader("Export experiments")
    experiments = api.experiments()
    if experiments:
        payload = []
        for experiment in experiments[:25]:
            try:
                payload.append(api.experiment(experiment["experiment_id"]))
            except Exception:
                payload.append(experiment)
        st.download_button(
            "Download experiments JSON",
            data=json.dumps(payload, indent=2, default=str),
            file_name="experiments.json",
            mime="application/json",
        )
        st.dataframe(experiments, use_container_width=True, hide_index=True)

    st.subheader("Export taxonomy and fault catalogue")
    st.download_button(
        "Download catalogue JSON",
        data=json.dumps(
            {"taxonomy": api.get("/taxonomy"), "faults": api.faults()}, indent=2, default=str
        ),
        file_name="catalogue.json",
        mime="application/json",
    )


main()