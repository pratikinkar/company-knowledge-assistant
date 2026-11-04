
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load company document
file_path = Path("documents/company_info.txt")

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()

print("Document loaded successfully!")

# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# 3. Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# 4. Store vectors in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vector_db",
    collection_name="company_knowledge"
)

print("Documents stored in ChromaDB successfully!")
print("Ingestion completed!")
