
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path


def ingest_pdf(pdf_path):

    # 1. Load PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        return "No readable text found in this PDF."

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # 3. Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Connect to existing ChromaDB
    vector_store = Chroma(
        collection_name="company_knowledge",
        embedding_function=embedding_model,
        persist_directory="./vector_db"
    )

    # 5. Store PDF chunks
    vector_store.add_documents(chunks)

    return f"Successfully added {len(chunks)} chunks from PDF!"
