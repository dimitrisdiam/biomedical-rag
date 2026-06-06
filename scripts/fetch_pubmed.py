# grab real abstracts from pubmed so the corpus isn't just the tiny sample.
# writes the same jsonl format, so point config.CORPUS_PATH at the output and
# rebuild the index.
# run: python -m scripts.fetch_pubmed --query "metformin type 2 diabetes" --max 200 --out data/pubmed_corpus.jsonl

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_ids(query: str, retmax: int, email: str) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "email": email}
    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    return data["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str], email: str) -> list[dict]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": email}
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        root = ET.parse(resp).getroot()

    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle") or ""
        abstract = " ".join(t.text or "" for t in article.findall(".//AbstractText"))
        if abstract.strip():
            docs.append({"id": f"pmid:{pmid}", "title": title, "text": abstract.strip()})
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument("--email", default="your-email@example.com")
    parser.add_argument("--out", default="data/pubmed_corpus.jsonl")
    args = parser.parse_args()

    ids = search_ids(args.query, args.max, args.email)
    print(f"Found {len(ids)} PubMed ids.")

    docs = []
    for i in range(0, len(ids), 100):  # efetch caps how many ids per call
        docs.extend(fetch_abstracts(ids[i:i + 100], args.email))
        time.sleep(0.5)  # don't hammer the api

    with open(args.out, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {len(docs)} abstracts to {args.out}")


if __name__ == "__main__":
    main()
