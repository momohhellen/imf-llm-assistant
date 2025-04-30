# imf-llm-assistant/app.py
import streamlit as st
from ingest import load_and_index_documents
from qa_engine import query_vector_store
from summarizer import summarize_text
import os
import tempfile

st.set_page_config(page_title="IMF LLM Assistant")
st.title("🌍 IMF LLM Assistant: Understand Policy Better")

# Upload IMF or World Bank report
uploaded_file = st.file_uploader("Upload a policy report (PDF or TXT):", type=["pdf", "txt"])

if uploaded_file:
    st.success("Document uploaded! Building index...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    load_and_index_documents(tmp_path)
    st.success("✅ Document indexed successfully!")

    st.subheader("🔍 Ask a question about this document")
    question = st.text_input("Your question:")

    if st.button("Ask") and question:
        answer = query_vector_store(question)
        st.markdown(f"**Answer:** {answer}")

        simplified = summarize_text(answer)
        st.markdown(f"**🧠 Simplified Explanation:** {simplified}")
