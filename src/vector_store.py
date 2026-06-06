# faiss wrapper. build, save, load, search

import json
from pathlib import Path

import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.Index:
    # inner product + normalised vectors = cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, chunks, index_path: Path, chunks_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def load_index(index_path: Path, chunks_path: Path):
    index = faiss.read_index(str(index_path))
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks


def search(index, query_vector: np.ndarray, k: int):
    scores, ids = index.search(query_vector, k)
    return list(zip(ids[0].tolist(), scores[0].tolist()))
