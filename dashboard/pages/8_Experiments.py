"""Experiment runner: reproducible configurations and executed results."""


import streamlit as st
from dashboard.common import client, guard, page


@guard
def main() -> None:
    page("Experiments", "🧪")
    api = client()

    st.subheader("Run a stored configuration")
    configs = api.experiment_configs()
    available = configs.get("available", [])
    if available:
        choice = st.selectbox("Configuration", available)
        if st.button("Run experiment", type="primary"):
            with st.spinner("Executing experiment..."):
                try:
                    result = api.run_experiment(name=choice)
                    st.session_state["last_experiment"] = result
                    st.success(f"Experiment `{result['name']}` completed ({result['duration_s']:.1f}s).")
                except Exception as exc:
                    st.error(str(exc))
        st.caption(choice)
    else:
        st.warning("No experiment configurations found in experiments/configs/.")

    st.subheader("Ad-hoc experiment")
    with st.form("adhoc_form"):
        fault_types = st.multiselect("Fault types", [f["fault_type"] for f in api.faults()])
        modes = st.multiselect(
            "Modes",
            ["mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"],
            default=["mode3_agentic_with_recovery"],
        )
        repetitions = st.number_input("Repetitions", 1, 10, 1)
        seed = st.number_input("Seed", 0, 10**6, 42)
        limit = st.number_input("Max questions", 1, 50, 6)
        submitted = st.form_submit_button("Run ad-hoc experiment")
    if submitted:
        payload = {
            "name": "adhoc",
            "modes": list(modes),
            "fault_types": list(fault_types),
            "repetitions": int(repetitions),
            "seed": int(seed),
            "limit": int(limit),
        }
        with st.spinner("Executing..."):
            try:
                result = api.run_experiment(**payload)
                st.session_state["last_experiment"] = result
                st.success("Completed.")
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get("last_experiment")
    if result:
        st.subheader("Latest result")
        for key in ("mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"):
            block = result.get(key)
            if not block:
                continue
            st.markdown(f"### {key}")
            st.json({k: v for k, v in block.items() if k != "per_query"}, expanded=False)
        if result.get("comparison"):
            st.markdown("### Comparison")
            st.json(result["comparison"], expanded=False)

    st.subheader("Experiment history")
    experiments = api.experiments()
    if experiments:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect experiment", [e["experiment_id"] for e in experiments])
        detail = api.experiment(selected)
        st.json({k: v for k, v in detail.items() if k not in {"metrics", "runs"}}, expanded=False)
        st.subheader("Metrics")
        st.json(detail.get("metrics", {}), expanded=False)
        st.subheader("Runs")
        st.dataframe(detail.get("runs", []), use_container_width=True, hide_index=True)


main()