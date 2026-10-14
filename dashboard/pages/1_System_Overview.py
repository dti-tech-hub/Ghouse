"""System overview: health, configuration, taxonomy and recovery strategies."""


import streamlit as st
from dashboard.common import client, guard, json_block, label_banner, page


@guard
def main() -> None:
    page("System Overview", "🛠️")
    label_banner()
    api = client()

    health = api.health()
    system = api.system()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", health["status"])
    col2.metric("Documents", health["documents"])
    col3.metric("Chunks", health["chunks"])
    col4.metric("Version", health["version"])

    st.subheader("Runtime configuration")
    left, right = st.columns(2)
    left.json(health, expanded=False)
    right.json(system.get("health", {}).get("settings", {}), expanded=False)

    st.subheader("Budgets (bounded control loop)")
    budget = system.get("health", {}).get("budget", {})
    if budget:
        columns = st.columns(len(budget))
        for column, (key, value) in zip(columns, budget.items()):
            column.metric(key.replace("_", " "), value)

    st.subheader("External evaluation adapters (optional)")
    st.json(system.get("external_evaluation", {}))

    st.subheader("Failure taxonomy")
    taxonomy = api.get("/taxonomy")
    rows = []
    for failure_class, subtypes in taxonomy.get("subtypes", {}).items():
        for subtype in subtypes:
            rows.append(
                {
                    "failure_class": failure_class,
                    "subtype": subtype["subtype"],
                    "stage": subtype["stage"],
                    "severity": subtype["severity"],
                    "recovery_actions": ", ".join(subtype["recovery_actions"]),
                    "description": subtype["description"],
                }
            )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Recovery strategies")
    st.json(api.get("/recovery/strategies"), expanded=False)

    st.subheader("Experiment configurations")
    st.json(system.get("experiment_configs", []))
    st.json({"database": health["database"], "labels": health["labels"]}, expanded=False)


main()