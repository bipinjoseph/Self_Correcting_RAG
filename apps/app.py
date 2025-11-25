import os
import sys
import streamlit as st

# ---------- Path setup so we can import from scripts/ ----------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from self_correcting_rag import self_correcting_rag


def main():
    st.set_page_config(page_title="Self-Correcting RAG", layout="wide")

    st.title("Self-Correcting RAG System")

    st.write(
        "Ask a question about your document. The system will retrieve relevant chunks, "
        "generate a draft answer, then fact-check and correct it."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=8, value=3)
        show_draft = st.checkbox("Show draft answer", value=True)
        show_chunks = st.checkbox("Show retrieved chunks", value=True)

    # Main input area
    st.subheader("Question")
    query = st.text_area("Enter your question:", height=120, key="user_query")

    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Thinking..."):
            try:
                draft_answer, final_answer, chunks = self_correcting_rag(query, top_k=top_k)
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
                return

        if show_draft:
            st.subheader("Draft Answer (before fact-check)")
            st.write(draft_answer)

        st.subheader("Final Answer (after self-correction)")
        st.write(final_answer)

        if show_chunks:
            st.subheader("Retrieved Chunks")
            for c in chunks:
                with st.expander(f"{c['chunk_id']} (distance={c['distance']:.4f})"):
                    st.write(c["text"])


if __name__ == "__main__":
    main()
