"""Query playground: the full agentic pipeline, stage by stage."""


import streamlit as st
from dashboard.common import client, guard, label_banner, page, table


@guard
def main() -> None:
    page("Query Playground", "🎮")
    label_banner()
    api = client()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Query")
        mode = st.selectbox(
            "Mode",
            ["mode3_agentic_with_recovery", "mode2_agentic_no_recovery", "mode1_normal_rag"],
            index=0,
        )
        query = st.text_area(
            "Question",
            value="Within how many minutes must a Severity 1 incident be acknowledged by the on-call engineer?",
            height=90,
        )
        top_k = st.slider("top_k", 1, 15, 5)
        method = st.selectbox("Retrieval method", ["hybrid", "vector", "bm25"], index=0)
        rerank = st.checkbox("Enable reranking", value=True)
        run = st.button("Run query", type="primary", use_container_width=True)

    with right:
        st.subheader("Demo questions")
        try:
            dataset = api.datasets()
            options = {q["question_id"]: q["question"] for q in dataset["demo_questions"]}
            choice = st.selectbox("Load a demo question", list(options))
            st.caption(options[choice])
            if st.button("Use this question"):
                st.session_state["_chosen_question"] = options[choice]
                st.session_state["_rerun"] = True
        except Exception as exc:
            st.warning(str(exc))

    st.subheader("Fault injection (optional)")
    faults = st.multiselect(
        "Inject faults",
        [f["fault_type"] for f in api.faults()],
        default=[],
        help="Leave empty for the healthy pipeline.",
    )

    chosen = st.session_state.pop("_chosen_question", None)
    if chosen:
        query = chosen
    trigger = run or st.session_state.pop("_rerun", False)
    if not trigger:
        st.info("Enter a question and press **Run query**.")
        return

    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "retrieval_method": method,
        "use_rerank": rerank,
        "faults": faults,
    }
    with st.spinner("Running the agentic pipeline..."):
        try:
            response = api.query(**payload)
        except Exception as exc:
            st.error(str(exc))
            return
    st.session_state["last_query_response"] = response

    if response["abstained"]:
        st.error("ABSTAINED — insufficient reliable evidence after bounded recovery.")
    elif response["recovered"]:
        st.success("RECOVERED — a repair was applied and verified before answering.")
    else:
        st.success("ANSWERED — no failure was detected.")

    tabs = st.tabs(
        [
            "1 Agent plan",
            "2 Retrieval",
            "3 Ranking",
            "4 Context",
            "5 Answer",
            "6 Signals",
            "7 Diagnosis",
            "8 Recovery",
            "9 Trace",
        ]
    )

    with tabs[0]:
        st.json(response["understanding"], expanded=True)
        st.markdown("**Sub-questions / sub-results**")
        st.dataframe(response["sub_results"], use_container_width=True, hide_index=True)

    with tabs[1]:
        diag = response["retrieval_diagnostics"]
        cols = st.columns(4)
        cols[0].metric("method", diag.get("method", "-"))
        cols[1].metric("returned", diag.get("returned", 0))
        cols[2].metric("top coverage", round(diag.get("top_overlap", 0.0), 3))
        cols[3].metric("useful ratio", round(diag.get("useful_ratio", 0.0), 3))
        st.json(diag, expanded=False)
        st.markdown("**Retrieved chunks**")
        st.dataframe(
            [
                {
                    "rank": r["rank"],
                    "score": round(r["score"], 4),
                    "dense": r.get("dense_score"),
                    "sparse": r.get("sparse_score"),
                    "rerank": r.get("rerank_score"),
                    "source": r["source"].split("/")[-1],
                    "page": r.get("page"),
                    "content": r["content"][:220],
                }
                for r in response["retrieved"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        st.json(response["signals"].get("ranking", {}), expanded=True)
        st.dataframe(
            [
                {
                    "rank": r["rank"],
                    "original_rank": r.get("original_rank"),
                    "score": round(r["score"], 4),
                    "rerank_score": r.get("rerank_score"),
                    "moved": (r.get("original_rank") or r["rank"]) != r["rank"],
                }
                for r in response["retrieved"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        context = response["context"]
        cols = st.columns(5)
        cols[0].metric("items", len(context.get("items", [])))
        cols[1].metric("tokens", context.get("token_estimate", 0))
        cols[2].metric("coverage", round(context.get("coverage", 0.0), 3))
        cols[3].metric("sources", context.get("source_diversity", 0))
        cols[4].metric("truncated", context.get("truncation_applied", False))
        for item in context.get("items", []):
            st.markdown(f"**rank {item['chunk'].get('rank')}** - `{item['chunk']['source'].split('/')[-1]}`")
            st.text(item["text"][:600])
            st.divider()

    with tabs[4]:
        st.markdown("### Final answer")
        st.write(response["answer"] or "_(empty)_")
        st.markdown("### Baseline answer (before recovery)")
        st.caption(response["baseline_answer"] or "_(none)_")
        st.json(response["generation"], expanded=False)

    with tabs[5]:
        st.json(response["signals"], expanded=True)
        rows = []
        for section, values in response["signals"].items():
            if isinstance(values, dict):
                for key, value in values.items():
                    rows.append({"section": section, "signal": key, "value": value})
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tabs[6]:
        if not response["diagnoses"]:
            st.info("No diagnosis was produced.")
        for diagnosis in response["diagnoses"]:
            st.markdown(
                f"**{diagnosis['failure_class']} / {diagnosis['subtype']}** "
                f"(tier `{diagnosis['tier']}`, confidence {diagnosis['confidence']:.2f})"
            )
            st.json(diagnosis, expanded=False)

    with tabs[7]:
        if not response["recoveries"]:
            st.info("No recovery attempt was made.")
        for attempt in response["recoveries"]:
            icon = {"success": "OK", "failure": "WARN", "skipped": "SKIP", "error": "ERR"}.get(
                attempt["status"], "-"
            )
            st.markdown(
                f"### {icon} attempt {attempt['attempt_index']} - `{attempt['action']}` ({attempt['status']})"
            )
            st.caption(attempt["rationale"])
            cols = st.columns(4)
            cols[0].metric("latency ms", round(attempt["latency_ms"], 2))
            cols[1].metric("added ms", round(attempt["added_latency_ms"], 2))
            cols[2].metric("tokens", attempt["tokens_used"])
            cols[3].metric("cost", attempt["cost"])
            st.json(attempt.get("verification", {}), expanded=False)
        if response.get("abstention"):
            st.error("Abstention record")
            st.json(response["abstention"], expanded=False)

    with tabs[8]:
        table(response["steps"])
        st.json(response["metrics"], expanded=False)


main()