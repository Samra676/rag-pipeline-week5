# rag-pipeline-week5
# RAG Pipeline for Machine Learning PDFs


## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using PDF documents. It extracts ML-related text, stores embeddings in FAISS, and generates answers to user queries using a QA model.


## Features
- Load and filter PDF content for ML keywords
- Generate embeddings with SentenceTransformers
- Store embeddings in FAISS for fast retrieval
- Question-Answering using DistilBERT
- Flask web app for user interaction
- JSON API endpoint `/ask` for programmatic queries


## Requirements
- Python 3.8+
- Libraries: `pypdf`, `sentence-transformers`, `faiss`, `transformers`, `flask`, `numpy`


## Installation
```bash
pip install pypdf sentence-transformers faiss-cpu transformers flask numpy
