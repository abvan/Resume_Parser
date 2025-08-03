from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Test text
text = "hi world"

# Get the embedding
embedding = model.encode(text)

print("Embedding length:", len(embedding))
print("First 10 dimensions:", embedding[:10])