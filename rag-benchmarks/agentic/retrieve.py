"""
Hybrid retrieval: BM25 + BGE-large dense embeddings with BGE cross-encoder reranking.

Loads pre-computed article embeddings and BM25 index. For each query batch,
BM25 and dense candidates are unioned then reranked with a cross-encoder
to produce the final top-n documents.

Supports parallel reranking across multiple GPUs via reranker_devices.

Used as a module by pipeline.py, not run directly.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridRetriever:
    """Loads corpus, embeddings, BM25 index, and reranker(s); serves batched queries."""

    def __init__(self, corpus_path: str, embeddings_path: str,
                 embedding_model: str, index_dir: str,
                 bm25_top_k: int, dense_top_k: int,
                 reranker_model: str, top_n: int,
                 reranker_devices: list[str] = None):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.top_n = top_n

        if reranker_devices is None:
            reranker_devices = ["cuda:2"]

        self.embedder = SentenceTransformer(embedding_model, device=reranker_devices[0])
        self.embeddings = np.load(embeddings_path)
        self.searcher = LuceneSearcher(index_dir)

        self.rerankers = [CrossEncoder(reranker_model, device=d) for d in reranker_devices]

        self.corpus_texts: list[str] = []
        self.corpus_dates: list[str] = []
        with open(corpus_path) as f:
            for line in f:
                rec = json.loads(line)
                self.corpus_texts.append(rec["contents"])
                self.corpus_dates.append(rec.get("publish_date", ""))

    def _rerank_chunk(self, reranker: CrossEncoder, pairs: list) -> np.ndarray:
        """Score pairs with a single reranker, sub-batched to control memory."""
        RERANK_BATCH = 512
        scores = np.empty(len(pairs))
        for start in range(0, len(pairs), RERANK_BATCH):
            batch = pairs[start:start + RERANK_BATCH]
            scores[start:start + len(batch)] = reranker.predict(batch)
        return scores

    def retrieve_batch(self, queries: list[str]) -> list[list[str]]:
        """Retrieve top-n documents for each query via BM25 + dense union + reranking.

        Splits reranking across available reranker GPUs for parallelism.
        """
        bm25_ids = [self._bm25_search(q) for q in queries]

        query_embs = self.embedder.encode(queries, show_progress_bar=False)
        scores_matrix = query_embs @ self.embeddings.T

        MAX_RERANK_CHARS = 32000
        all_pairs = []
        group_sizes = []
        all_candidate_ids = []

        for i, query in enumerate(queries):
            dense_ids = list(np.argsort(scores_matrix[i])[::-1][:self.dense_top_k])

            seen = set()
            candidate_ids = []
            for idx in bm25_ids[i] + dense_ids:
                if idx not in seen:
                    seen.add(idx)
                    candidate_ids.append(idx)

            all_candidate_ids.append(candidate_ids)
            pairs = [(query, self.corpus_texts[j][:MAX_RERANK_CHARS]) for j in candidate_ids]
            all_pairs.extend(pairs)
            group_sizes.append(len(pairs))

        # Parallel reranking across GPUs
        n_rerankers = len(self.rerankers)
        if n_rerankers == 1:
            all_scores = self._rerank_chunk(self.rerankers[0], all_pairs)
        else:
            chunk_size = (len(all_pairs) + n_rerankers - 1) // n_rerankers
            with ThreadPoolExecutor(max_workers=n_rerankers) as pool:
                futures = []
                for r_idx in range(n_rerankers):
                    start = r_idx * chunk_size
                    end = min(start + chunk_size, len(all_pairs))
                    if start < end:
                        futures.append(pool.submit(
                            self._rerank_chunk, self.rerankers[r_idx],
                            all_pairs[start:end],
                        ))
                all_scores = np.concatenate([f.result() for f in futures])

        # Unpack scores per query
        results = []
        offset = 0
        for i, size in enumerate(group_sizes):
            scores = all_scores[offset:offset + size]
            offset += size
            candidate_ids = all_candidate_ids[i]

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
