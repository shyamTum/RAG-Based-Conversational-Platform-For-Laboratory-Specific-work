import os
import json
import time
import argparse
from typing import List, Dict, Any

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

KB_DIR = r".\kb_store"

MODEL_DIR = r".\sentence-transformer\sentence-transformer\all-MiniLM-L6-v2"

MIXTRAL_BASE_URL = "http://mb-qs-pp-a100.na.pg.com:5621"
MIXTRAL_API_KEY = "MIXTRAL_API_KEY"
MIXTRAL_MODEL = "mixtral"

TOP_K_DEFAULT = 6
MAX_CONTEXT_CHARS = 12000
TIMEOUT_SEC = 180
RETRIES = 2

def paths():
    return {
        "index": os.path.join(KB_DIR, "faiss.index"),
        "meta": os.path.join(KB_DIR, "meta.jsonl"),
    }


def load_meta() -> List[Dict[str, Any]]:
    rows = []
    with open(paths()["meta"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index() -> faiss.Index:
    return faiss.read_index(paths()["index"])


def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    v = embedder.encode([q], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(v)
    return v


def retrieve(index: faiss.Index, meta_rows: List[Dict[str, Any]], embedder: SentenceTransformer, query: str, k: int):
    qv = embed_query(embedder, query)
    scores, idxs = index.search(qv, k)
    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= idx < len(meta_rows):
            row = dict(meta_rows[idx])
            row["score"] = float(score)
            hits.append(row)
    return hits


def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    total = 0
    for h in hits:
        src = h.get("source", "unknown")
        cid = h.get("chunk_id", -1)
        txt = (h.get("text") or "").strip()
        block = f"[SOURCE: {src} | chunk {cid}]\n{txt}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)


def call_mixtral(question: str, context: str, temperature: float, max_tokens: int) -> str:
    url = MIXTRAL_BASE_URL.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIXTRAL_API_KEY}",
    }
    system = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and correctly."

    payload = {
        "model": MIXTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
            if r.status_code != 200:
                return f"Mixtral error {r.status_code}: {r.text}"
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout:
            last_err = f"ReadTimeout after {TIMEOUT_SEC}s (attempt {attempt+1}/{RETRIES+1})"
        except Exception as e:
            last_err = f"Error calling Mixtral: {e}"
        time.sleep(1.5 * (attempt + 1))

    return last_err or "Unknown error calling Mixtral."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", "--query", dest="query", required=True)
    ap.add_argument("--k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--show_sources", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"MODEL_DIR not found: {MODEL_DIR}")

    if not os.path.exists(paths()["index"]) or not os.path.exists(paths()["meta"]):
        raise SystemExit("KB not found. Run: python ingest_kb_2.py")

    embedder = SentenceTransformer(MODEL_DIR)
    index = load_index()
    meta_rows = load_meta()

    hits = retrieve(index, meta_rows, embedder, args.query, args.k)
    if args.show_sources:
        print("\n=== TOP HITS ===")
        for h in hits:
            print(f"- {h.get('source')} (chunk {h.get('chunk_id')}) score={h.get('score'):.3f}")

    context = build_context(hits)
    if not context.strip():
        print("\nNo context retrieved. Increase --k or ingest more docs.")
        return

    ans = call_mixtral(args.query, context, args.temp, args.max_tokens)
    print("\n=== ANSWER ===\n")
    print(ans)


if __name__ == "__main__":
    main()