"""Knowledge base: documents, ingestion and index health."""


import streamlit as st
from dashboard.common import client, guard, page


@guard
def main() -> None:
    page("Knowledge Base", "📚")
    api = client()
    documents = api.documents()

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(documents))
    col2.metric("Chunks", sum(d["chunk_count"] for d in documents))
    col3.metric("Characters", sum(d["chars"] for d in documents))

    st.subheader("Ingest")
    with st.form("ingest_form"):
        path = st.text_input("Path (server-side directory or file)", value="data/demo_documents")
        force = st.checkbox("Re-ingest even if unchanged", value=False)
        submitted = st.form_submit_button("Ingest")
    if submitted:
        with st.spinner("Ingesting and rebuilding indexes..."):
            try:
                result = api.ingest(path, force=force)
                st.success(f"Indexed {result['documents']} document(s) into {result['chunks']} chunk(s).")
                if result.get("errors"):
                    st.error("\n".join(result["errors"]))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Documents")
    if documents:
        st.dataframe(documents, use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect document", [d["document_id"] for d in documents])
        document = api.get(f"/documents/{selected}")
        st.json({k: v for k, v in document.items() if k != "chunks"}, expanded=False)
        st.subheader("Chunks")
        st.dataframe(document["chunks"], use_container_width=True, hide_index=True)
    else:
        st.warning("The knowledge base is empty. Ingest data/demo_documents above.")


main()