# loading the corpus and cutting documents into chunks

import json
from pathlib import Path
from typing import Iterator


def load_corpus(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    # overlap the windows, otherwise a fact sitting on a chunk boundary gets
    # split in two and you can never retrieve it
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    for doc in docs:
        body = f"{doc.get('title', '')}. {doc.get('text', '')}".strip()
        for i, piece in enumerate(chunk_text(body, size, overlap)):
            chunks.append({
                "chunk_id": f"{doc['id']}::{i}",
                "doc_id": doc["id"],  # keep the source doc so eval can check it
                "title": doc.get("title", ""),
                "text": piece,
            })
    return chunks


def iter_chunks(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
