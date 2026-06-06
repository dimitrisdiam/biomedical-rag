# Biomedical RAG with Evaluation

A retrieval augmented generation system that answers questions about biomedical
papers, using real abstracts pulled from PubMed. I built it mostly to do the part
people usually skip: actually checking whether it works, both whether it finds the
right documents and whether the answers stick to them.

## The idea

A RAG system can break in two places. The retriever can grab the wrong documents,
or the model can ignore the documents and answer from memory, which gives you a
confident wrong answer. On medical text the second one is the dangerous case, so I
added a small evaluation step that measures both instead of just showing the cases
where it happens to work.

## What it does

1. Loads documents and splits them into overlapping chunks.
2. Embeds the chunks with a sentence transformer and puts them in a FAISS index.
3. For a question, it pulls the closest chunks and asks the model to answer using
   only those.
4. Scores retrieval and answers on a small labelled question set.

Retrieval and evaluation run with no API key. The answer step uses the OpenAI API
if a key is set, and otherwise returns the top passage, so you can still run and
score the whole thing offline.

## Layout

```
biomedical-rag/
├── config.py                  paths, model names, chunk size, top k
├── data/
│   ├── sample_corpus.jsonl    small corpus so it runs out of the box
│   └── eval_questions.jsonl   questions with the doc ids and facts I expect
├── src/
│   ├── ingest.py              load and chunk
│   ├── embed.py               embeddings
│   ├── vector_store.py        faiss build/save/load/search
│   ├── llm.py                 llm call with an offline fallback
│   └── rag.py                 retrieve, build the prompt, answer
├── scripts/
│   ├── build_index.py         build the index
│   ├── ask.py                 ask from the terminal
│   └── fetch_pubmed.py        pull real abstracts to scale up
├── eval/
│   ├── evaluate_retrieval.py  hit@k and mrr
│   └── evaluate_answers.py    keyword recall and groundedness
└── tests/
    └── test_pipeline.py       chunking and ranking
```

## Running it

```bash
pip install -r requirements.txt

# build the index from the bundled corpus
python -m scripts.build_index

# ask something (set OPENAI_API_KEY first for a real answer)
python -m scripts.ask --question "How does metformin lower blood sugar?"

# score it
python -m eval.evaluate_retrieval
python -m eval.evaluate_answers
```

To use real data, fetch a bigger corpus and point config.CORPUS_PATH at it:

```bash
python -m scripts.fetch_pubmed --query "metformin type 2 diabetes" --max 200 --out data/pubmed_corpus.jsonl
```

## How I evaluate it

Retrieval:

- hit@k: did a relevant document land in the top k.
- mrr: one over the rank of the first relevant document, averaged.

Answers:

- keyword recall: did the facts I expect show up in the answer.
- groundedness: how much of the answer is backed by the retrieved text. A low
  score means the model is probably answering from memory, not the documents.

## Results

Numbers from my run go here. Format looks like:

```
hit@4: 1.000
MRR:   0.917

Mean keyword recall: 0.86
Mean groundedness:   0.78
```

Then a line or two on what stood out, like where retrieval missed or where the
answer drifted past the context.

## Notes

- The bundled corpus is tiny on purpose so the repo runs without downloading
  anything big. fetch_pubmed.py is for running it on real abstracts.
- Groundedness is token overlap right now, which is why it runs offline. Next step
  is an llm judge that scores each claim, and a reranker on top of the first
  retrieval pass.
