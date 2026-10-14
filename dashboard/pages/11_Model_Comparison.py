"""Model comparison: registered models, providers and retrieval configuration."""


import streamlit as st
from dashboard.common import client, guard, page


@guard
def main() -> None:
    page("Model Comparison", "🧾")
    api = client()
    system = api.system()
    config = api.get("/config")

    st.subheader("Registered models")
    st.dataframe(api.get("/models").get("models", []), use_container_width=True, hide_index=True)

    st.subheader("Configured generation")
    st.json(config["settings"]["generation"], expanded=True)

    st.subheader("Available providers")
    st.markdown(
        """
| provider | requires | notes |
|---|---|---|
| `mock` | nothing | deterministic and offline; results are labelled DEMO / MOCK RESULTS |
| `ollama` | a local Ollama daemon | free, fully local |
| `openai` | `OPENAI_API_KEY` | also works against any OpenAI-compatible server |
| `claude` | `ANTHROPIC_API_KEY` | optional adapter |

To switch, set `generation.provider` / `generation.model` in `configs/default.yaml`
or export `ARR_LLM_PROVIDER` / `ARR_LLM_MODEL` before starting the backend.
        """
    )

    st.subheader("Retrieval configuration")
    st.json(config["settings"]["retrieval"], expanded=True)

    st.subheader("Embedding configuration")
    st.json(config["settings"]["retrieval"], expanded=False)
    st.caption(
        "The offline default is a deterministic hashed TF-IDF projector. Set "
        "`retrieval.embedding_backend: sentence_transformer` for a neural encoder."
    )

    st.subheader("External evaluation adapters")
    st.json(system.get("external_evaluation", {}))
    st.caption(
        "Reported metrics always come from the deterministic custom metrics; "
        "RAGAS / DeepEval are optional cross-checks."
    )


main()