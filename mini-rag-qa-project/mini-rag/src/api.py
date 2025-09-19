# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.search import Retriever
import os

app = FastAPI(title="Mini RAG Reranker")

# Configure paths
FAISS_PATH = "outputs/faiss.index"
ID_MAP = "outputs/id_map.json"
BM25_PATH = "outputs/bm25.pkl"
DB_PATH = "data/chunks.db"

retriever = Retriever(FAISS_PATH, ID_MAP, BM25_PATH, DB_PATH)

class AskRequest(BaseModel):
    q: str
    k: Optional[int] = 3
    mode: Optional[str] = "hybrid"  # "baseline" or "hybrid"
    abstain_threshold: Optional[float] = 0.35  # tune

@app.post("/ask")
def ask(req: AskRequest):
    if req.mode == "baseline":
        contexts = retriever.baseline_search(req.q, k=req.k)
        # pick top context as answer (extractive)
        top_score = contexts[0]["score"] if contexts else 0
        if top_score < req.abstain_threshold:
            return {"answer": None, "contexts": contexts, "reranker_used": "baseline"}
        ans = contexts[0]["text"][:800]  # short extract
        return {"answer": ans, "contexts": contexts, "reranker_used": "baseline"}
    else:
        contexts = retriever.hybrid_rerank(req.q, k=req.k)
        top_score = contexts[0]["final_score"] if contexts else 0
        if top_score < req.abstain_threshold:
            return {"answer": None, "contexts": contexts, "reranker_used": "hybrid"}
        ans = contexts[0]["text"][:800]
        return {"answer": ans, "contexts": contexts, "reranker_used": "hybrid"}
 
