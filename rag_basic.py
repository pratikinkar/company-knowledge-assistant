
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 1. Load the company document
file_path = Path("documents/company_info.txt")
document = file_path.read_text(encoding="utf-8")

# 2. Split the document into small chunks
chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]

print(f"Document loaded. Total chunks: {len(chunks)}")

# 3. Ask a question
question = input("\nAsk a question about the company: ")

# 4. Find the most relevant chunk using keyword matching
def tokenize(text):
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

stop_words = {
    "the", "is", "a", "an", "in", "of", "to",
    "for", "was", "what", "how", "much", "does",
    "did", "and", "are", "company", "tell", "me",
    "about", "please"
}

question_words = tokenize(question) - stop_words

scored_chunks = [] 

for chunk in chunks:
    chunk_words = tokenize(chunk)
    score = len(question_words & chunk_words)
    scored_chunks.append((score, chunk))

scored_chunks.sort(key=lambda item: item[0], reverse=True)

# Take up to 3 relevant chunks
retrieved = [
    chunk
    for score, chunk in scored_chunks[:3]
    if score > 0
]

if not retrieved:
    print("\nNo relevant information found in the document.")
    raise SystemExit

context = "\n".join(retrieved)

print("\nRetrieved information:")
print(context)

# 5. Give the retrieved information to the LLM
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

prompt = f"""
You are a company information assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nAI Answer:")
print(response.content)
