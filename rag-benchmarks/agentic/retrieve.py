"""
Hybrid retrieval: BM25 + BGE-large dense embeddings with BGE cross-encoder reranking.

Loads pre-computed article embeddings and BM25 index. For each query batch,
BM25 and dense candidates are unioned then reranked with a cross-encoder
to produce the final top-n documents.

Used as a module by pipeline.py, not run directly.
"""

import json

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridRetriever:
    """Loads corpus, embeddings, BM25 index, and reranker once; serves batched queries."""

    def __init__(self, corpus_path: str, embeddings_path: str,
                 embedding_model: str, index_dir: str,
                 bm25_top_k: int, dense_top_k: int,
                 reranker_model: str, top_n: int):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.top_n = top_n

        self.embedder = SentenceTransformer(embedding_model, device="cuda")
        self.embeddings = np.load(embeddings_path)
        self.searcher = LuceneSearcher(index_dir)
        self.reranker = CrossEncoder(reranker_model, device="cuda")

        self.corpus_texts: list[str] = []
        self.corpus_dates: list[str] = []
        with open(corpus_path) as f:
            for line in f:
                rec = json.loads(line)
                self.corpus_texts.append(rec["contents"])
                self.corpus_dates.append(rec.get("publish_date", ""))

    def retrieve_batch(self, queries: list[str]) -> list[list[str]]:
        """Retrieve top-n documents for each query via BM25 + dense union + reranking.

        Returns a list of length len(queries), each element a list of
        top_n document texts ordered by descending reranker score.
        """
        bm25_ids = [self._bm25_search(q) for q in queries]

        query_embs = self.embedder.encode(queries, show_progress_bar=False)
        scores_matrix = query_embs @ self.embeddings.T

        results = []
        for i, query in enumerate(queries):
            dense_ids = list(np.argsort(scores_matrix[i])[::-1][:self.dense_top_k])

            seen: set[int] = set()
            candidate_ids: list[int] = []
            for idx in bm25_ids[i] + dense_ids:
                if idx not in seen:
                    seen.add(idx)
                    candidate_ids.append(idx)

            candidate_texts = [self.corpus_texts[j] for j in candidate_ids]

            MAX_RERANK_CHARS = 32000
            scores = self.reranker.predict(
                [(query, t[:MAX_RERANK_CHARS]) for t in candidate_texts]
            )
            ranked = sorted(zip(scores, candidate_ids), reverse=True)

            formatted = []
            for _, idx in ranked[:self.top_n]:
                date = self.corpus_dates[idx]
                text = self.corpus_texts[idx]
                if date:
                    date = date.split("T")[0]
                    formatted.append(f"[Published: {date}]\n{text}")
                else:
                    formatted.append(text)
            results.append(formatted)

        return results

    def _bm25_search(self, query: str) -> list[int]:
        """BM25 search; returns corpus indices in BM25 rank order."""
        hits = self.searcher.search(query, k=self.bm25_top_k)
        return [int(hit.docid) for hit in hits]
