
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY missing!")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embedding_model,
    persist_directory="./vector_db"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)


def ask_question(question):

    results = vector_store.similarity_search_with_score(
        question,
        k=2
    )

    if not results:
        return "I could not find this information in the document."

    context = "\n".join(
        doc.page_content for doc, score in results
    )

    prompt = f"""
You are a company information assistant.

Answer using ONLY the context below.
Do not invent facts or numbers.
If the answer is unavailable, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
