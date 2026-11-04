
import streamlit as st
import tempfile
import os

from rag_pipeline import ask_question
from pdf_ingest import ingest_pdf

st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Company Knowledge Assistant")
st.caption("AI-powered document question answering using RAG")

# PDF upload section
st.sidebar.header("Upload Company Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.sidebar.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                result = ingest_pdf(temp_path)
                st.sidebar.success(result)

            except Exception as e:
                st.sidebar.error(f"PDF processing failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User question
question = st.chat_input("Ask a question about your documents...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):
            answer = ask_question(question)
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
