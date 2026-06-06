# check the retriever. each question lists the doc ids that actually answer it,
# and I measure whether those come back in the top k.
#   hit@k: did a relevant doc show up in the top k
#   mrr:   1 / rank of the first relevant doc, averaged
# if retrieval is bad the answer is doomed no matter how good the prompt is.
# runs offline, no key needed.
# run: python -m eval.evaluate_retrieval

import json

import config
from src.embed import Embedder
from src.llm import LLMClient
from src.rag import RagPipeline
from src import vector_store


def load_eval(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reciprocal_rank(retrieved_doc_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main(k: int = config.TOP_K) -> None:
    index, chunks = vector_store.load_index(config.INDEX_PATH, config.CHUNKS_PATH)
    pipeline = RagPipeline(index, chunks, Embedder(config.EMBED_MODEL), LLMClient(config.LLM_MODEL), k)

    questions = load_eval(config.EVAL_PATH)
    hits, rr_total = 0, 0.0

    for q in questions:
        relevant = set(q["relevant_doc_ids"])
        retrieved = pipeline.retrieve(q["question"], k)
        # one doc can produce several chunks, so dedupe but keep the order
        ordered_docs = list(dict.fromkeys(r.doc_id for r in retrieved))

        hit = any(d in relevant for d in ordered_docs)
        hits += int(hit)
        rr_total += reciprocal_rank(ordered_docs, relevant)

        status = "hit " if hit else "miss"
        print(f"[{status}] {q['question']}  ->  {ordered_docs}")

    n = len(questions)
    print(f"\nQuestions: {n}")
    print(f"hit@{k}: {hits / n:.3f}")
    print(f"MRR:    {rr_total / n:.3f}")


if __name__ == "__main__":
    main()
