"""
End-to-end vanilla RAG pipeline: retrieve → rerank → generate.

Loads config, initializes all components, and runs inference on a JSONL
file of questions. Can also be imported and used programmatically.

Usage:
    python pipeline.py --config ../config.yaml --questions ../../eval_sample/all.jsonl
"""

import argparse
import json
import os

import yaml
from tqdm import tqdm

from retrieve import HybridRetriever
from rerank import Reranker
from generate import Generator


def load_questions(path: str) -> list[dict]:
    """Load questions from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_pipeline(retriever: HybridRetriever, reranker: Reranker,
                 generator: Generator, questions: list[dict]) -> list[dict]:
    """Run retrieval + reranking for all questions, then batch generate."""
    docs_per_question = []
    for q in tqdm(questions, desc="Retrieving and reranking"):
        candidates = retriever.retrieve(q["question"])
        reranked = reranker.rerank(q["question"], candidates)
        docs_per_question.append(reranked)

    print(f"Generating answers for {len(questions)} questions...")
    answers = generator.generate_batch(
        [q["question"] for q in questions],
        docs_per_question,
    )

    results = []
    for q, answer in zip(questions, answers):
        results.append({
            "question": q["question"],
            "expected_answer": q.get("answer", q.get("expected_answer", "")),
            "model_answer": answer,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["generation"]["model"]
    if model_cfg.startswith("${") and model_cfg.endswith("}"):
        cfg["generation"]["model"] = os.environ[model_cfg[2:-1]]

    artifacts_dir = cfg["artifacts_dir"]

    print("Loading retriever...")
    retriever = HybridRetriever(
        index_dir=os.path.join(artifacts_dir, "index"),
        corpus_path=os.path.join(artifacts_dir, "corpus", "corpus.jsonl"),
        embeddings_path=os.path.join(artifacts_dir, "embeddings.npy"),
        embedding_model=cfg["embeddings"]["model"],
        bm25_top_k=cfg["retrieval"]["bm25_top_k"],
        dense_top_k=cfg["retrieval"]["dense_top_k"],
    )

    print("Loading reranker...")
    reranker = Reranker(
        model_name=cfg["reranking"]["model"],
        top_n=cfg["reranking"]["top_n"],
    )

    print("Loading generator...")
    gen_cfg = cfg["generation"]
    generator = Generator(
        model=gen_cfg["model"],
        tp=gen_cfg["tp"],
        max_model_len=gen_cfg["max_model_len"],
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
    )

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    results = run_pipeline(retriever, reranker, generator, questions)

    output_path = args.output or args.questions.replace(".jsonl", "_rag_results.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
