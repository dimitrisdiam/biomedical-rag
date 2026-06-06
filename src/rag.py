# the actual rag bit: pull the relevant chunks, then answer using only those

from dataclasses import dataclass

from src.embed import Embedder
from src.llm import LLMClient
from src import vector_store

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about biomedical literature. "
    "Answer only from the provided context. If the context does not contain the "
    "answer, say you do not know. Keep answers short and factual."
)


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class RagPipeline:
    def __init__(self, index, chunks, embedder: Embedder, llm: LLMClient, top_k: int):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, question: str, k: int | None = None) -> list[Retrieved]:
        k = k or self.top_k
        query_vector = self.embedder.encode([question])
        hits = vector_store.search(self.index, query_vector, k)
        results = []
        for idx, score in hits:
            if idx < 0:  # faiss returns -1 when it has fewer than k results
                continue
            chunk = self.chunks[idx]
            results.append(Retrieved(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                title=chunk["title"],
                text=chunk["text"],
                score=float(score),
            ))
        return results

    def build_user_prompt(self, question: str, contexts: list[Retrieved]) -> str:
        # tag each block with its doc id so the model can point at sources
        blocks = [f"[{c.doc_id}] {c.text}" for c in contexts]
        joined = "\n\n".join(blocks)
        return f"Question: {question}\n\nContext:\n{joined}"

    def answer(self, question: str) -> dict:
        contexts = self.retrieve(question)
        user_prompt = self.build_user_prompt(question, contexts)
        answer = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": [c.doc_id for c in contexts],
            "contexts": contexts,
        }
