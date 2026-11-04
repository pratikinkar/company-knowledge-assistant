
from sentence_transformers import SentenceTransformer

# Load a local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The company earned 600000 rupees in September.",
    "The business made six lakh rupees last month.",
    "The analytics department has 12 employees.",
    "The weather is sunny today."
]

# Convert sentences into vectors
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

# Compare the first two sentences
similarity = model.similarity(
    embeddings[0:1],
    embeddings[1:2]
)

print("\nSimilarity between sentence 1 and 2:")
print(float(similarity[0][0]))
 