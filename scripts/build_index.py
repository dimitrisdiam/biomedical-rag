# build the index from the corpus
# run: python -m scripts.build_index

import config
from src.ingest import load_corpus, build_chunks
from src.embed import Embedder
from src import vector_store


def main() -> None:
    docs = load_corpus(config.CORPUS_PATH)
    chunks = build_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Loaded {len(docs)} documents, {len(chunks)} chunks.")

    embedder = Embedder(config.EMBED_MODEL)
    vectors = embedder.encode([c["text"] for c in chunks])

    index = vector_store.build_index(vectors)
    vector_store.save_index(index, chunks, config.INDEX_PATH, config.CHUNKS_PATH)
    print(f"Index written to {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
