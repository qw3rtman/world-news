"""
Cross-encoder reranking: score (question, document) pairs and keep top-n.

Takes the merged candidates from hybrid retrieval and re-scores each pair
using bge-reranker-v2-m3, returning the highest-scoring documents.

Used as a module by pipeline.py, not run directly.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """Loads cross-encoder once, reranks per query."""

    def __init__(self, model_name: str, top_n: int):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, question: str, docs: list[dict]) -> list[dict]:
        """Score each document against the question, return top-n."""
        MAX_RERANK_CHARS = 32000
        pairs = [[question, doc["contents"][:MAX_RERANK_CHARS]] for doc in docs]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self.top_n]]
