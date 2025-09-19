# src/build_index.py
import sqlite3
import argparse
import faiss
import pickle
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

def build_faiss_and_bm25(db_path, faiss_path, bm25_path, id_map_path):
    # Connect to SQLite where chunks are stored
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get chunk_id + text (so ids match DB)
    cursor.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("❌ No chunks found in database. Run ingest.py first.")
        return

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    # ---- FAISS Index ----
    print("🔹 Building FAISS index with embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(embeddings)

    faiss.write_index(index, faiss_path)
    print(f"✅ FAISS index saved to {faiss_path}")

    # ---- BM25 Index ----
    print("🔹 Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_path, "wb") as f:
        pickle.dump((bm25, tokenized), f)
    print(f"✅ BM25 index saved to {bm25_path}")

    # ---- ID Map ---- (just a list of chunk_ids in FAISS order)
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print(f"✅ ID map saved to {id_map_path}")
    print(f"📦 Indexed {len(ids)} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from chunks DB")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite chunks.db")
    parser.add_argument("--faiss_path", type=str, required=True, help="Path to save FAISS index")
    parser.add_argument("--bm25_path", type=str, required=True, help="Path to save BM25 index")
    parser.add_argument("--id_map", type=str, required=True, help="Path to save ID map JSON")
    args = parser.parse_args()

    build_faiss_and_bm25(args.db_path, args.faiss_path, args.bm25_path, args.id_map)
