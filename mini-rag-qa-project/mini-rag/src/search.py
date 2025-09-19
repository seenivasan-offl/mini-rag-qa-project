# src/search.py
import json, pickle, sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

MODEL_NAME = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, faiss_path, id_map_path, bm25_path, db_path):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(faiss_path)

        # Load ID map
        with open(id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)  # list of chunk_ids

        # Load BM25 + tokenized corpus (tuple)
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, tuple):  # new format
            self.bm25, self.tokenized = data
        else:  # fallback if dict
            self.bm25 = data["bm25"]
            self.tokenized = data["tokenized"]

        self.db_path = db_path

    def _fetch_chunks(self, chunk_ids):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        res = []
        for cid in chunk_ids:
            row = c.execute(
                "SELECT chunk_id, doc_name, title, source_url, chunk_index, text FROM chunks WHERE chunk_id=?",
                (cid,)
            ).fetchone()
            if row:
                res.append({
                    "chunk_id": row[0],
                    "doc_name": row[1],
                    "title": row[2],
                    "source_url": row[3],
                    "chunk_index": row[4],
                    "text": row[5]
                })
        conn.close()
        return res

    def embed_query(self, q):
        q_emb = self.model.encode([q], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        return q_emb

    def baseline_search(self, q, k=5):
        q_emb = self.embed_query(q)
        D, I = self.index.search(q_emb, k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        chunk_ids = [self.id_map[i] for i in idxs]
        chunks = self._fetch_chunks(chunk_ids)
        for c, s in zip(chunks, scores):
            c["score"] = float(s)
        return chunks

    def hybrid_rerank(self, q, k=5, alpha=0.6, candidate_multiplier=4):
        # get more candidates from vector search then rerank
        cand_k = min(self.index.ntotal, k * candidate_multiplier)
        q_emb = self.embed_query(q)
        D, I = self.index.search(q_emb, cand_k)
        vec_scores = D[0]
        idxs = I[0]
        chunk_ids = [self.id_map[i] for i in idxs]
        chunks = self._fetch_chunks(chunk_ids)

        # BM25 scores
        tokens = q.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)

        # normalize vector scores
        vs = vec_scores
        vmin, vmax = float(vs.min()), float(vs.max())
        vnorm = (vs - vmin) / (vmax - vmin + 1e-8)

        # normalize bm25 scores for same candidates
        b_scores = np.array([bm25_scores[self.id_map.index(cid)] for cid in chunk_ids])
        bmin, bmax = float(b_scores.min()), float(b_scores.max())
        bnorm = (b_scores - bmin) / (bmax - bmin + 1e-8)

        # final score
        final_scores = alpha * vnorm + (1 - alpha) * bnorm

        results = []
        for i, cid in enumerate(chunk_ids):
            results.append({
                "chunk_id": cid,
                "text": chunks[i]["text"],
                "doc_name": chunks[i]["doc_name"],
                "title": chunks[i]["title"],
                "source_url": chunks[i]["source_url"],
                "vec_score": float(vec_scores[i]),
                "bm25_score": float(b_scores[i]),
                "final_score": float(final_scores[i])
            })

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return results[:k]
