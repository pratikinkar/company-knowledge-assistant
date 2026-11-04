
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY missing in .env file")

# 2. Load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Load company document
file_path = Path("documents/company_info.txt")
document = file_path.read_text(encoding="utf-8")

chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]

# 4. Create document embeddings
print("Creating document embeddings...")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True
)

# 5. Connect Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

print("\nSemantic RAG Assistant Ready!")

# 6. Ask questions continuously
while True:

    question = input(
        "\nAsk a question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    # 7. Convert question into embedding
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    # 8. Find relevant document information
    scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    top_indices = scores.argsort(descending=True)[:3]

    retrieved_chunks = []

    for index in top_indices:
        i = int(index.item())
        score = scores[i].item()

        if score >= 0.35:
            retrieved_chunks.append(chunks[i])

    # 9. Check whether relevant information exists
    if not retrieved_chunks:
        print("\nAI Answer:")
        print("I could not find this information in the document.")
        continue

    context = "\n".join(retrieved_chunks)

    print("\nRetrieved Context:")
    print(context)

    # 10. Send context and question to LLM
    prompt = f"""
You are a company information assistant.

Answer the user's question using ONLY the
information provided in the context.

Rules:
1. Do not invent facts or numbers.
2. If the answer is not available in the context,
   say: "I could not find this information in the document."
3. Give a clear and simple answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAI Answer:")
    print(response.content)
