"""Fault injection console with declared ground truth."""


import streamlit as st
import json

from dashboard.common import client, guard, page, table


@guard
def main() -> None:
    page("Fault Injection", "💉")
    api = client()
    faults = api.faults()

    st.subheader("Catalogue")
    table(faults)

    st.subheader("Inject a fault")
    left, right = st.columns(2)
    with left:
        selected = st.selectbox("Fault type", [f["fault_type"] for f in faults])
        definition = next(f for f in faults if f["fault_type"] == selected)
        st.markdown(f"**{definition['title']}**")
        st.caption(definition["description"])
        st.markdown(f"- target stage: `{definition['target_stage']}`")
        st.markdown(
            f"- expected: `{definition['expected_failure_class']}` / `{definition['expected_subtype']}`"
        )
        st.markdown(f"- symptom: {definition['expected_symptom']}")
        parameters = st.text_area("Parameters (JSON)", value="{}", height=80)
    with right:
        query = st.text_area(
            "Question (optional)", value="How often are vendors that process customer data audited?"
        )
        if st.button("Inject and run", type="primary", use_container_width=True):
            try:
                params = json.loads(parameters or "{}")
            except ValueError as exc:
                st.error(f"parameters must be valid JSON: {exc}")
                params = {}
            if not isinstance(params, dict):
                st.error("parameters must be a JSON object")
            else:
                with st.spinner("Injecting fault and running the pipeline..."):
                    try:
                        response = api.inject(selected, params, query or None)
                        st.session_state["last_injection"] = response
                        st.success(f"Injected `{selected}`.")
                    except Exception as exc:
                        st.error(str(exc))

    response = st.session_state.get("last_injection")
    if not response:
        return
    st.subheader("Injected fault (ground truth)")
    st.json({k: v for k, v in response.items() if k != "result"}, expanded=True)
    result = response.get("result")
    if not result:
        return
    first = result["diagnoses"][0] if result["diagnoses"] else {}
    cols = st.columns(4)
    cols[0].metric("detected class", first.get("failure_class", "-"))
    cols[1].metric("subtype", first.get("subtype", "-"))
    cols[2].metric("recovered", result["recovered"])
    cols[3].metric("abstained", result["abstained"])
    st.markdown("### Answer")
    st.write(result["answer"])
    if result["diagnoses"]:
        st.json(result["diagnoses"][0], expanded=False)
    if result["recoveries"]:
        table(result["recoveries"])


main()