# ask a question from the terminal
# run: python -m scripts.ask --question "What is metformin used for?"

import argparse

import config
from src.embed import Embedder
from src.llm import LLMClient
from src.rag import RagPipeline
from src import vector_store


def load_pipeline() -> RagPipeline:
    index, chunks = vector_store.load_index(config.INDEX_PATH, config.CHUNKS_PATH)
    embedder = Embedder(config.EMBED_MODEL)
    llm = LLMClient(config.LLM_MODEL)
    return RagPipeline(index, chunks, embedder, llm, config.TOP_K)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    pipeline = load_pipeline()
    result = pipeline.answer(args.question)

    print("\nAnswer:")
    print(result["answer"])
    # dict.fromkeys to drop duplicate doc ids but keep their order
    print("\nSources:", ", ".join(dict.fromkeys(result["sources"])))


if __name__ == "__main__":
    main()
