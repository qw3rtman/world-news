"""
Offline preprocessing: build corpus from valid articles, filter pre-computed
embeddings, and build BM25 index.

No chunking: articles are small (~1K tokens median, 2K max) so each article
is one document. Pre-computed BGE-large embeddings are filtered to valid
hashes and reordered to match the corpus JSONL.

Run once before inference:
    python preprocess.py --config config.yaml
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def load_valid_articles(articles_dir: str, valid_hashes: list[str]) -> list[dict]:
    """Load article text and metadata for each valid hash."""
    articles_dir = Path(articles_dir)
    articles = []

    for h in tqdm(valid_hashes, desc="Loading articles"):
        article_path = articles_dir / h / "article.txt"
        if not article_path.exists():
            continue

        text = article_path.read_text(errors="replace").strip()
        if len(text) < 100:
            continue

        # Read metadata for publish date
        publish_date = ""
        meta_path = articles_dir / h / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            sources = meta.get("sources", [])
            if sources:
                publish_date = sources[0].get("published_date", "")

        articles.append({
            "hash": h,
            "contents": text,
            "publish_date": publish_date,
        })

    return articles


def filter_embeddings(articles: list[dict], meta_path: str,
                      emb_path: str) -> np.ndarray:
    """Filter and reorder pre-computed embeddings to match corpus order.

    The pre-computed embeddings cover all 1473 articles (valid + invalidated).
    We select only the rows corresponding to valid articles, in corpus order.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    hash_to_idx = {h: i for i, h in enumerate(meta["hashes"])}
    all_embs = np.load(emb_path)

    indices = []
    for article in articles:
        idx = hash_to_idx.get(article["hash"])
        if idx is None:
            raise ValueError(f"Hash {article['hash']} not found in embeddings metadata")
        indices.append(idx)

    return all_embs[indices]


def write_corpus(articles: list[dict], corpus_dir: str):
    """Write articles as JSONL for Pyserini and dense retrieval."""
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_path = os.path.join(corpus_dir, "corpus.jsonl")

    with open(corpus_path, "w") as f:
        for i, article in enumerate(articles):
            record = {
                "id": str(i),
                "contents": article["contents"],
                "publish_date": article["publish_date"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(articles)} articles to {corpus_path}")


def build_bm25_index(corpus_dir: str, index_dir: str, threads: int = 4):
    """Build Pyserini Lucene index over the corpus JSONL."""
    os.makedirs(index_dir, exist_ok=True)

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    print(f"Building BM25 index: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Index built at {index_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load valid hashes
    with open(cfg["valid_hashes_file"]) as f:
        valid_hashes = [h.strip() for h in f if h.strip()]
    print(f"Loaded {len(valid_hashes)} valid hashes")

    # Load articles
    articles = load_valid_articles(cfg["articles_dir"], valid_hashes)
    print(f"Loaded {len(articles)} valid articles")

    artifacts_dir = cfg["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # Write corpus
    corpus_dir = os.path.join(artifacts_dir, "corpus")
    write_corpus(articles, corpus_dir)

    # Filter and save embeddings
    print("Filtering pre-computed embeddings to valid articles...")
    embeddings = filter_embeddings(
        articles,
        meta_path=cfg["embeddings"]["precomputed_meta"],
        emb_path=cfg["embeddings"]["precomputed_path"],
    )
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} to {emb_path}")

    # Build BM25 index
    index_dir = os.path.join(artifacts_dir, "index")
    build_bm25_index(corpus_dir, index_dir)

    print(f"\nPreprocessing complete. Artifacts in {artifacts_dir}/")


if __name__ == "__main__":
    main()
