"""
Hybrid retrieval: BM25 (Pyserini) + dense (BGE-large) with union and deduplication.

Given a question, retrieves candidates from both a Lucene index and a dense
embedding index, merges them, and returns unique documents for reranking.

Used as a module by pipeline.py, not run directly.
"""

import json

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """Loads BM25 index and dense embeddings once, serves queries."""

    def __init__(self, index_dir: str, corpus_path: str, embeddings_path: str,
                 embedding_model: str, bm25_top_k: int, dense_top_k: int):
        self.searcher = LuceneSearcher(index_dir)
        self.bm25_top_k = bm25_top_k

        self.embedder = SentenceTransformer(embedding_model)
        self.embeddings = np.load(embeddings_path)
        self.dense_top_k = dense_top_k

        self.corpus = []
        with open(corpus_path) as f:
            for line in f:
                self.corpus.append(json.loads(line))

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve documents via BM25 + dense, union and deduplicate."""
        bm25_results = self._bm25_search(question)
        dense_results = self._dense_search(question)

        seen = set()
        merged = []
        for doc in bm25_results + dense_results:
            if doc["id"] not in seen:
                seen.add(doc["id"])
                merged.append(doc)

        return merged

    def _bm25_search(self, question: str) -> list[dict]:
        """BM25 retrieval via Pyserini."""
        hits = self.searcher.search(question, k=self.bm25_top_k)
        return [self.corpus[int(hit.docid)] for hit in hits]

    def _dense_search(self, question: str) -> list[dict]:
        """Dense retrieval via cosine similarity against pre-computed embeddings."""
        query_emb = self.embedder.encode(question)
        scores = self.embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:self.dense_top_k]
        return [self.corpus[idx] for idx in top_indices]
