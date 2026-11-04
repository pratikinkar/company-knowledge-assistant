
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

file_path = Path("documents/company_info.txt")
document = file_path.read_text(encoding="utf-8")

chunks = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]

print("Creating document embeddings...")

chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)

print("\nSemantic Search Ready!")

while True:
    question = input(
        "\nAsk a question (or type 'exit' to stop): "
    ).strip()

    if question.lower() == "exit":
        print("Goodbye, Bhai!")
        break

    if not question:
        print("Please enter a question.")
        continue

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    top_indices = scores.argsort(descending=True)[:3]

    print("\nMost relevant information:\n")

    for index in top_indices:
        i = int(index.item())
        print(f"Score: {scores[i].item():.4f}")
        print(f"Text: {chunks[i]}\n")
