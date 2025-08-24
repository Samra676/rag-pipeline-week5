from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from flask import Flask, request, jsonify, render_template, flash
import os

# Flask app
app = Flask(__name__)
app.secret_key = "rag_secret_key"  # Required for flash messages

# Step 1: PDF load karo
pdf_path = "machine_learning.pdf"
reader = PdfReader(pdf_path)

# Filter only ML-related paragraphs
keywords = ["machine learning", "ML", "supervised", "unsupervised", "deep learning", "neural network"]
documents = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        text = text.strip()
        if any(kw.lower() in text.lower() for kw in keywords):
            documents.append(text)

if not documents:
    raise ValueError("PDF is empty or no ML-related text extracted!")

# Step 2: Embeddings (local + fallback to HuggingFace)
model_path = "models/all-MiniLM-L6-v2"
if os.path.exists(model_path):
    embedding_model = SentenceTransformer(model_path)
else:
    print("⚠ Local embedding model nahi mila. HuggingFace se download ho raha hai...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Step 3: FAISS index
dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(doc_embeddings)

print("✅ FAISS index bana liya")
print("📄 Total ML documents indexed:", faiss_index.ntotal)

# Step 4: QA model (local + fallback to HuggingFace)
qa_model_path = "models/distilbert-base-cased-distilled-squad"
if os.path.exists(qa_model_path):
    qa_pipeline = pipeline("question-answering", model=qa_model_path)
else:
    print("⚠ Local QA model nahi mila. HuggingFace se download ho raha hai...")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Step 5: RAG pipeline function
def rag_pipeline(query, max_sentences=3):
    if not query.strip():
        return "Please enter a valid question."

    # Retrieve context from FAISS
    D, I = faiss_index.search(embedding_model.encode([query]), k=3)
    retrieved_contexts = [documents[i] for i in I[0]]
    retrieved_context = " ".join(retrieved_contexts)

    if not retrieved_context.strip():
        return "No relevant ML information found in the documents."

    # Use QA model for concise answer
    result = qa_pipeline(question=query, context=retrieved_context)
    answer = result.get('answer', 'Unable to generate answer.')

    # Limit answer length
    sentences = answer.split('. ')
    short_answer = '. '.join(sentences[:max_sentences]).strip()
    if not short_answer.endswith('.'):
        short_answer += '.'

    return short_answer

# Step 6: API route (JSON)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    answer = rag_pipeline(query)
    return jsonify({"query": query, "answer": answer})

# Step 7: HTML form route
@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        query = request.form.get("query", "")
        if not query:
            flash("Please enter a question before submitting!")
        else:
            answer = rag_pipeline(query)
    return render_template("index.html", answer=answer)

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
