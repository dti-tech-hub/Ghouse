"""Recovery page: bounded loop, verification results and abstentions."""


import streamlit as st
from dashboard.common import client, guard, page, table


@guard
def main() -> None:
    page("Recovery", "🛠")
    api = client()
    traces = api.traces(limit=200)
    run_ids = [t["run_id"] for t in traces] or []
    if not run_ids:
        st.warning("No runs recorded yet.")
        return

    selected = st.selectbox("Run", run_ids)
    trace = api.trace(selected)
    recoveries = trace.get("recoveries", [])
    if not recoveries:
        st.info("This run needed no recovery.")
    else:
        table(recoveries, columns=[
            "attempt", "action", "status", "re_run_stage",
            "added_latency_ms", "verified", "skipped_stages",
        ])
        st.subheader("Verification detail")
        for attempt in recoveries:
            verification = attempt.get("verification", {})
            st.markdown(f"**{attempt['action']} -> {attempt['status']}** (passed: {verification.get('passed')})")
            table(
                verification.get("criteria", []),
                columns=["criterion", "passed", "observed", "required", "detail"],
            )

    st.subheader("Abstentions")
    abstentions = api.abstentions(limit=100)
    if abstentions:
        table(abstentions, columns=[
            "run_id", "reason", "attempted_recoveries", "attempts_used",
            "max_attempts", "message",
        ])
    else:
        st.info("No abstention recorded.")

    st.subheader("Aggregate metrics")
    st.json(api.metrics().get("aggregates", {}), expanded=False)


main()