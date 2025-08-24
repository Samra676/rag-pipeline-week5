from sentence_transformers import SentenceTransformer

# Model download hoga aur cache me save ho jayega
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model successfully downloaded and cached.")
