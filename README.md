# 📚 LLM Boot Camp – Week 4: Retrieval-Augmented Generation (RAG) with arXiv Papers

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using recent **arXiv cs.CL (Computation and Language)** research papers. The goal is to build a semantic search system that allows users to query a private knowledge base of scientific literature and retrieve relevant text passages for downstream use in AI agents.

This foundational system will evolve into a hybrid database in Week 5.

---

## 🎯 Learning Objectives

- Understand the components of a **Retriever-Reader QA pipeline**.
- Explore effective **document chunking strategies** (sliding window vs. section-based).
- Generate **dense vector embeddings** using Sentence Transformers.
- Index embeddings using **FAISS** for efficient similarity search.
- Serve results via a **FastAPI endpoint** for real-time querying.

---

## 🛠️ Project Design Overview

The RAG pipeline consists of the following stages:

1. **Data Collection**: Download 50 recent papers from `arXiv cs.CL`.
2. **Text Extraction**: Extract raw text from PDFs using `PyMuPDF`.
3. **Text Chunking**: Split documents into meaningful chunks (≤ 512 tokens) using a sliding window.
4. **Embedding Generation**: Encode chunks into 384-dimensional vectors using `all-MiniLM-L6-v2`.
5. **Indexing**: Build a **FAISS GPU-accelerated index** for fast similarity search.
6. **Query Interface**: Implement a **FastAPI** `/search` endpoint to retrieve top-3 relevant passages.
7. **Evaluation**: Test retrieval quality with sample queries and analyze results.

---

## 📁 Project Structure
