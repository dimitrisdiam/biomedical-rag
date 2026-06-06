# tests for chunking and retrieval ranking. uses a fake bag of words embedder so
# the suite doesn't have to download the real model

import numpy as np

from src.ingest import chunk_text, build_chunks
from src.rag import RagPipeline
from src import vector_store


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text("one two three", size=10, overlap=2)
    assert chunks == ["one two three"]


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, size=4, overlap=2)
    # last 2 words of one chunk should be the first 2 of the next
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_build_chunks_keeps_doc_id():
    docs = [{"id": "d1", "title": "T", "text": "a b c d e f"}]
    chunks = build_chunks(docs, size=3, overlap=1)
    assert all(c["doc_id"] == "d1" for c in chunks)


class BagOfWordsEmbedder:
    # cheap stand-in for the real embedder, just so I can check the relevant
    # chunk comes back first
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, texts):
        vecs = []
        for t in texts:
            words = set(t.lower().split())
            v = np.array([1.0 if w in words else 0.0 for w in self.vocab], dtype="float32")
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm else v)
        return np.array(vecs, dtype="float32")


def test_retriever_ranks_relevant_chunk_first():
    chunks = [
        {"chunk_id": "a::0", "doc_id": "a", "title": "", "text": "metformin lowers glucose in the liver"},
        {"chunk_id": "b::0", "doc_id": "b", "title": "", "text": "statins lower cholesterol in the blood"},
    ]
    vocab = ["metformin", "glucose", "liver", "statins", "cholesterol", "blood", "lowers", "lower"]
    embedder = BagOfWordsEmbedder(vocab)
    vectors = embedder.encode([c["text"] for c in chunks])
    index = vector_store.build_index(vectors)

    pipeline = RagPipeline(index, chunks, embedder, llm=None, top_k=2)
    results = pipeline.retrieve("how does metformin affect glucose")
    assert results[0].doc_id == "a"
