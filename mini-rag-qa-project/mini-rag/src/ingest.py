# src/ingest.py
import argparse, json, sqlite3, pathlib
import pdfplumber
from tqdm import tqdm
import random, numpy as np

random.seed(42)
np.random.seed(42)

MAX_CHARS = 800  # approx paragraph-size; tune if needed


def chunk_paragraph(p, max_chars=MAX_CHARS):
    p = p.strip()
    if not p:
        return []
    if len(p) <= max_chars:
        return [p]
    # simple sliding window split (keeps reproducible)
    return [p[i:i + max_chars].strip() for i in range(0, len(p), max_chars)]


def extract_text_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            txt = pg.extract_text()
            if txt:
                pages.append(txt)
    return "\n\n".join(pages)


def main(pdf_dir, db_path, sources_path):
    pdf_dir = pathlib.Path(pdf_dir)
    db_path = pathlib.Path(db_path)

    # --- FIX: handle list format in sources.json ---
    with open(sources_path, "r", encoding="utf-8") as f:
        sources_list = json.load(f)

    sources = {}
    for entry in sources_list:
        title = entry["title"]
        url = entry.get("url", "")
        # Guess filename: take last part after slash if present
        filename = pathlib.Path(url).name
        if not filename.endswith(".pdf"):
            filename = f"{title}.pdf"
        sources[filename] = {"title": title, "url": url}

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT,
        title TEXT,
        source_url TEXT,
        chunk_index INTEGER,
        text TEXT
      )
    """)
    conn.commit()

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        doc_name = pdf_path.name
        meta = sources.get(doc_name, {"title": doc_name, "url": ""})
        title = meta.get("title", doc_name)
        url = meta.get("url", "")

        text = extract_text_from_pdf(pdf_path)
        paragraphs = [p for p in (s.strip() for s in text.split("\n\n")) if p]
        chunk_index = 0
        for para in paragraphs:
            chunks = chunk_paragraph(para)
            for ch in chunks:
                c.execute(
                    "INSERT INTO chunks (doc_name, title, source_url, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
                    (doc_name, title, url, chunk_index, ch)
                )
                chunk_index += 1
    conn.commit()
    conn.close()
    print("Ingestion complete. DB:", db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--db_path", default="data/chunks.db")
    parser.add_argument("--sources", default="data/sources.json")
    args = parser.parse_args()
    main(args.pdf_dir, args.db_path, args.sources)
