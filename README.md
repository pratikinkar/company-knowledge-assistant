
# Company Knowledge Assistant

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG). Users can upload PDF documents and ask questions in natural language to receive context-aware answers.

## Features

- PDF document upload and processing
- Text extraction and intelligent document chunking
- Semantic search using Hugging Face embeddings
- Vector storage and retrieval using ChromaDB
- Context-aware question answering using Groq LLM
- Interactive chat interface built with Streamlit
- Answers based on retrieved document content
- Handles questions when information is unavailable in the uploaded documents

## Tech Stack

- **Programming Language:** Python
- **LLM Integration:** Groq
- **Framework:** LangChain
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Database:** ChromaDB
- **PDF Processing:** PyPDF
- **Frontend:** Streamlit

## Project Architecture

1. Upload a PDF document.
2. Extract text from the document.
3. Split the text into smaller chunks.
4. Generate vector embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks using semantic search.
7. Send the retrieved context and user question to the Groq LLM.
8. Display the generated answer in the Streamlit interface.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/pratikinkar/company-knowledge-assistant.git
cd company-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U langchain langchain-groq langchain-community langchain-huggingface langchain-chroma langchain-text-splitters chromadb sentence-transformers python-dotenv streamlit pypdf
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never publish your actual API key.

### 5. Ingest the sample document

```bash
python ingest.py
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

## Project Structure

```text
company-knowledge-assistant/
│
├── documents/
│   └── company_info.txt
│
├── app.py
├── rag_pipeline.py
├── pdf_ingest.py
├── ingest.py
├── main.py
├── rag_basic.py
├── rag_semantic.py
├── semantic_search.py
├── embedding_test.py
├── read_document.py
├── .gitignore
└── README.md
```

## Future Improvements

- Support multiple document formats
- Display document sources and page references
- Document management and deletion
- Improve retrieval accuracy
- Deploy the application online

## Author

**Pratik Inkar**

GitHub: https://github.com/pratikinkar
Live Demo: https://company-knowledge-assistant-l5uxul8micudas2xdezblq.streamlit.app/

---

*Note: The included company information is synthetic demonstration data for learning and testing purposes.*
=======
