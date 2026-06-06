# check the answers, two things:
#   keyword recall - do the facts I expect actually show up in the answer
#   groundedness   - how much of the answer is backed by the retrieved text.
#                    low groundedness usually means the model made something up.
# groundedness here is just token overlap so it runs offline. a stricter version
# would use an llm to judge each claim (noted in the readme).
# run: python -m eval.evaluate_answers

import json
import re

import config
from src.embed import Embedder
from src.llm import LLMClient
from src.rag import RagPipeline
from src import vector_store


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def keyword_recall(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected if kw.lower() in answer_lower)
    return found / len(expected)


def groundedness(answer: str, contexts: list[str]) -> float:
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in contexts:
        context_tokens |= tokenize(c)
    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)


def main() -> None:
    index, chunks = vector_store.load_index(config.INDEX_PATH, config.CHUNKS_PATH)
    llm = LLMClient(config.LLM_MODEL)
    pipeline = RagPipeline(index, chunks, Embedder(config.EMBED_MODEL), llm, config.TOP_K)

    with open(config.EVAL_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    recall_total, ground_total = 0.0, 0.0
    if not llm.online:
        # without a key the answers come from the fallback, so these numbers
        # are only a rough sanity check
        print("Note: no OPENAI_API_KEY, using the offline fallback, scores are indicative only.\n")

    for q in questions:
        result = pipeline.answer(q["question"])
        context_texts = [c.text for c in result["contexts"]]

        recall = keyword_recall(result["answer"], q.get("expected_keywords", []))
        ground = groundedness(result["answer"], context_texts)
        recall_total += recall
        ground_total += ground

        print(f"Q: {q['question']}")
        print(f"   keyword recall: {recall:.2f}   groundedness: {ground:.2f}")

    n = len(questions)
    print(f"\nMean keyword recall: {recall_total / n:.3f}")
    print(f"Mean groundedness:   {ground_total / n:.3f}")


if __name__ == "__main__":
    main()
