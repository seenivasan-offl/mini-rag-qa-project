Mini-RAG Q&A: Industrial & Machine Safety
Overview

This project implements a mini Retrieval-Augmented Generation (RAG) question-answering system over a small set of industrial safety PDFs. The system retrieves relevant chunks from the documents using FAISS embeddings and BM25 keyword matching, and then ranks them using a hybrid reranker to provide accurate, extractive answers with citations.

The goal is to demonstrate a small, reproducible Q&A system with baseline similarity search and a hybrid reranker, suitable for research or internship purposes.

Dataset

20 public PDFs on industrial and machine safety (stored in data/)

sources.json keeps metadata: title + source URL

Documents are split into paragraph-sized chunks and stored in chunks.db

Setup

Clone the repository

git clone <your-github-repo-url>
cd mini-rag


Create virtual environment and install dependencies

python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt


Build indexes

python src/build_index.py --db_path data/chunks.db --faiss_path outputs/faiss.index --bm25_path outputs/bm25.pkl --id_map outputs/id_map.json


Start API server

uvicorn src.api:app --reload

Usage

Send a POST request to /ask endpoint:

curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"q":"machine safety rules","k":5,"mode":"hybrid"}'


Request parameters:

q – Question string

k – Number of top results to retrieve

mode – hybrid (vector + BM25) or vector (FAISS only)

Response:

answer – Extractive answer from documents

contexts – List of retrieved chunks with scores and sources

reranker_used – Indicates whether hybrid reranker was applied

Results

The system was tested using 8 questions (see questions.json). Sample output table:

Question	Answer	Reranker Used	Source
What is Personal Protective Equipment (PPE)?	Ring they know the Standard Operating Procedures (SOPs)...	hybrid	Hokuyo Safety Guide

What is lockout-tagout and why is it used?	Power up is to remove the energy from the system...	hybrid	Rockwell SafeBook

How should machine guards be designed to prevent entanglement?	Providing a barrier synchronized with the operating cycle...	hybrid	OSHA Publication 3170

...	...	...	...

(Full results are in results.json)

Learnings

Hybrid reranker improves accuracy: Combining vector similarity with BM25 keyword matching ensures more relevant chunks rise to the top.

Chunking is crucial: Properly splitting PDFs into paragraph-sized chunks significantly affects retrieval quality.

Abstaining when unsure: Implemented a simple score threshold to avoid giving incorrect answers.

Extractive answers with citations: Ensures the answers are grounded in actual documents, which is essential for safety-related queries.

Example Questions

What is Personal Protective Equipment (PPE)?

What is lockout-tagout and why is it used?

How should machine guards be designed to prevent entanglement?

What are common causes of industrial fires in machine shops?

How should you respond to hydraulic fluid leaks?

What is the role of Material Safety Data Sheets (MSDS)?

When should emergency stop buttons be used on machinery?

How to safely perform maintenance on rotating equipment?

Notes

CPU-only, no paid APIs were used.

Repeatable results: Random seeds are set for embeddings and retrieval.

Outputs are extractive and cited.
