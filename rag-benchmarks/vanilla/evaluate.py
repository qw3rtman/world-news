"""
Evaluate the vanilla RAG pipeline on all question types.

Loads eval_sample/{task}.jsonl files, runs retrieval → reranking → generation,
then scores with a 3-way LLM judge (MATCH / NONMATCH / REFUSAL).

Question types: direct, temporal, compositional_2hop, compositional_3hop,
compositional_4hop, boundary_abstention, indexing.

Usage:
    cd rag-benchmarks/vanilla
    python evaluate.py --config ../config.yaml
"""

import argparse
import json
import os

import yaml
from vllm import SamplingParams

from retrieve import HybridRetriever
from rerank import Reranker
from generate import Generator
from pipeline import run_pipeline


JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, a reference answer, and a model's response, classify the model's response into exactly one category.

Question: {question}
Reference answer: {expected_answer}
Model response: {model_answer}

Categories:
- MATCH: The model's response contains the key factual claim from the reference answer. Different wording is fine, but the core fact must be present and correct. Partial correctness does not count.
- NONMATCH: The model's response is factually wrong, vague, off-topic, or missing the key claim from the reference answer.
- REFUSAL: The model explicitly declines to answer, says it doesn't know, lacks information, or cannot answer the question.

Think and then reply with exactly one word: MATCH, NONMATCH, or REFUSAL."""


TASK_FILES = [
    "direct.jsonl",
    "temporal.jsonl",
    "compositional_2hop.jsonl",
    "compositional_3hop.jsonl",
    "compositional_4hop.jsonl",
    "boundary_abstention.jsonl",
    "indexing.jsonl",
]


def load_eval_questions(eval_dir: str) -> list[dict]:
    """Load all question types from eval_sample/, normalizing answer fields."""
    questions = []
    for filename in TASK_FILES:
        path = os.path.join(eval_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue

        with open(path) as f:
            for line in f:
                q = json.loads(line)
                task = q.get("task", filename.replace(".jsonl", ""))
                q["split"] = task

                # Normalize answer field across question types
                if "answer_after_update" in q:
                    q["answer"] = q["answer_after_update"]
                elif "gold_answer" in q:
                    q["answer"] = q["gold_answer"]
                # else: "answer" field already present (direct, temporal, indexing)

                q["expected_answer"] = q.get("answer", "")
                questions.append(q)

        count = sum(1 for qq in questions if qq["split"] == task)
        print(f"  {filename}: {count} questions")

    return questions


def parse_verdict(text: str) -> str:
    """Parse LLM judge output to extract MATCH/NONMATCH/REFUSAL verdict."""
    import re
    text = text.strip().upper()
    for word in reversed(text.split()):
        clean = word.strip(".,!?:;\"'()")
        if clean in ("MATCH", "NONMATCH", "REFUSAL"):
            return clean.lower()
    if re.search(r"\bNONMATCH\b", text):
        return "nonmatch"
    if re.search(r"\bREFUSAL\b", text):
        return "refusal"
    if re.search(r"\bMATCH\b", text) and not re.search(r"\bNON\s*MATCH\b", text):
        return "match"
    return "nonmatch"


def run_judge_api(results: list[dict], model_spec: str) -> list[dict]:
    """Score results using an API model (e.g. openai:gpt-5.4-mini)."""
    from openai import OpenAI

    provider, model_id = model_spec.split(":", 1)
    if provider == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "deepinfra":
        client = OpenAI(
            api_key=os.environ.get("DEEPINFRA_API_KEY", ""),
            base_url="https://api.deepinfra.com/v1/openai",
        )
    else:
        raise ValueError(f"Unknown API provider: {provider}")

    for r in results:
        prompt = JUDGE_PROMPT.format(
            question=r["question"],
            expected_answer=r.get("expected_answer", r.get("answer", "")),
            model_answer=r["model_answer"],
        )
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=128,
        )
        r["judge_verdict"] = parse_verdict(response.choices[0].message.content or "")
    return results


def run_judge_vllm(llm, tokenizer, results: list[dict]) -> list[dict]:
    """Score results using a local vLLM model."""
    def _apply_chat(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    prompts = [
        _apply_chat(JUDGE_PROMPT.format(
            question=r["question"],
            expected_answer=r.get("expected_answer", r.get("answer", "")),
            model_answer=r["model_answer"],
        ))
        for r in results
    ]
    params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate(prompts, params)
    for r, o in zip(results, outputs):
        r["judge_verdict"] = parse_verdict(o.outputs[0].text)
    return results


def print_summary(results: list[dict]):
    """Print per-task pass@8 / allcorrect@8 breakdown."""
    splits = sorted(set(r["split"] for r in results))
    total_n = len(results)
    total_any = sum(1 for r in results if r["any_correct"])
    total_all = sum(1 for r in results if r["all_correct"])

    for split in splits:
        sr = [r for r in results if r["split"] == split]
        n = len(sr)
        any_c = sum(1 for r in sr if r["any_correct"])
        all_c = sum(1 for r in sr if r["all_correct"])
        print(f"\n{split} (N={n}):")
        print(f"  pass@8:       {any_c:4d} ({100*any_c/n:5.1f}%)")
        print(f"  allcorrect@8: {all_c:4d} ({100*all_c/n:5.1f}%)")

    if total_n:
        print(f"\nOverall (N={total_n}): pass@8={total_any/total_n*100:.1f}%, "
              f"allcorrect@8={total_all/total_n*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: use gold article(s) as context instead of retrieval")
    parser.add_argument("--judge-model", type=str, default="openai:gpt-5.4-mini",
                        help="Judge model: 'openai:MODEL' for API, or local path for vLLM")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["generation"]["model"]
    if model_cfg.startswith("${") and model_cfg.endswith("}"):
        cfg["generation"]["model"] = os.environ[model_cfg[2:-1]]

    artifacts_dir = cfg["artifacts_dir"]
    eval_dir = cfg["eval_dir"]

    # Load all question types
    print("Loading evaluation questions...")
    all_questions = load_eval_questions(eval_dir)
    print(f"Loaded {len(all_questions)} total questions")

    if args.oracle:
        # Oracle mode: provide gold article(s) as context per question
        from pathlib import Path
        articles_dir = cfg["articles_dir"]
        markets_dir = os.path.join(os.path.dirname(articles_dir.rstrip("/")), "markets")

        # Load pre-fetched Wikipedia context for compositional hops
        wiki_context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "wikipedia_oracle_context.json")
        _wiki_context = {}
        if os.path.exists(wiki_context_path):
            with open(wiki_context_path) as f:
                _wiki_context = json.load(f)
            print(f"Loaded {len(_wiki_context)} Wikipedia oracle contexts")

        docs_per_question = []
        n_matched = 0
        for q in all_questions:
            gold_docs = []
            split = q.get("split", "")

            if split in ("direct", "temporal"):
                # hash -> articles/{hash}/article.txt
                h = q.get("hash", "")
                if h:
                    article_path = os.path.join(articles_dir, h, "article.txt")
                    if os.path.exists(article_path):
                        text = open(article_path).read().strip()
                        if text:
                            gold_docs.append({"contents": text, "publish_date": ""})

            elif split == "indexing":
                # market_id + spike_id -> spike metadata -> article hashes
                mid = q.get("market_id", "")
                sid = q.get("spike_id", "")
                if mid and sid is not None:
                    spike_meta_path = os.path.join(markets_dir, str(mid), f"spike_{sid}", "metadata.json")
                    if os.path.exists(spike_meta_path):
                        import json as _json
                        spike_meta = _json.load(open(spike_meta_path))
                        for article_info in spike_meta.get("articles", []):
                            af = article_info.get("file", "")
                            if af:
                                h = af.replace(".txt", "")
                                article_path = os.path.join(articles_dir, h, "article.txt")
                                if os.path.exists(article_path):
                                    text = open(article_path).read().strip()
                                    if text:
                                        gold_docs.append({"contents": text, "publish_date": ""})

            elif split.startswith("compositional"):
                # hop 1: article from corpus
                af = q.get("article_file", "")
                if af:
                    parts = af.split("/")
                    for i, p in enumerate(parts):
                        if p == "articles" and i + 1 < len(parts):
                            h = parts[i + 1]
                            article_path = os.path.join(articles_dir, h, "article.txt")
                            if os.path.exists(article_path):
                                text = open(article_path).read().strip()
                                if text:
                                    gold_docs.append({"contents": text, "publish_date": ""})
                            break

                # hops 2+: Wikipedia context from pre-fetched file
                for hop in q.get("chain", []):
                    src = hop.get("source", "")
                    if src.startswith("wikipedia:"):
                        page = src.replace("wikipedia:", "").replace(" ", "_")
                        fact = hop.get("fact", "")
                        wiki_key = f"{page}||{fact}"
                        wiki_text = _wiki_context.get(wiki_key, "")
                        if wiki_text:
                            gold_docs.append({"contents": wiki_text, "publish_date": ""})

            # boundary_abstention: no oracle (answer is "not in corpus")
            # — gold_docs stays empty, model must say "I don't know"

            # For oracle: prepend instruction that news article is most recent
            if gold_docs and split.startswith("compositional"):
                gold_docs[0]["contents"] = (
                    "[NOTE: This news article contains the MOST RECENT information. "
                    "When it conflicts with Wikipedia background, the news article is correct.]\n\n"
                    + gold_docs[0]["contents"]
                )

            if gold_docs:
                n_matched += 1
            docs_per_question.append(gold_docs)

        print(f"Oracle: {n_matched}/{len(all_questions)} questions matched to gold articles")

    else:
        # Standard retrieval: BM25 + dense + rerank
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

        from tqdm import tqdm
        print(f"Retrieving and reranking {len(all_questions)} questions...")
        docs_per_question = []
        for q in tqdm(all_questions, desc="Retrieving and reranking"):
            candidates = retriever.retrieve(q["question"])
            reranked = reranker.rerank(q["question"], candidates)
            docs_per_question.append(reranked)

        import gc, torch
        del reranker, retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Freed retriever/reranker memory")

    # Generation
    print("Loading generator...")
    gen_cfg = cfg["generation"]
    n_trials = gen_cfg.get("n_trials", 8)
    generator = Generator(
        model=gen_cfg["model"],
        tp=gen_cfg["tp"],
        max_model_len=gen_cfg["max_model_len"],
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg.get("top_p", 0.8),
        top_k=gen_cfg.get("top_k", 20),
        min_p=gen_cfg.get("min_p", 0),
        n_trials=n_trials,
    )

    print(f"Generating {n_trials} answers per question for {len(all_questions)} questions...")
    all_predictions = generator.generate_batch(
        [q["question"] for q in all_questions],
        docs_per_question,
    )

    # Judge all predictions (flatten, judge, reshape)
    print(f"\nRunning LLM judge ({args.judge_model}) on {len(all_questions)} × {n_trials} predictions...")
    flat_results = []
    for q, preds in zip(all_questions, all_predictions):
        for pred in preds:
            flat_results.append({
                "question": q["question"],
                "expected_answer": q.get("expected_answer", q.get("answer", "")),
                "model_answer": pred,
            })

    if ":" in args.judge_model:
        flat_results = run_judge_api(flat_results, args.judge_model)
    else:
        flat_results = run_judge_vllm(generator.llm, generator.tokenizer, flat_results)

    # Reshape and compute pass@8 / allcorrect@8
    results = []
    idx = 0
    for q, preds in zip(all_questions, all_predictions):
        verdicts = [flat_results[idx + j]["judge_verdict"] for j in range(len(preds))]
        idx += len(preds)
        n_correct = sum(1 for v in verdicts if v == "match")
        out = {
            **q,
            "predictions": preds,
            "verdicts": verdicts,
            "n_correct": n_correct,
            "any_correct": n_correct > 0,
            "all_correct": n_correct == len(preds),
            "n_trials": len(preds),
            "condition": "oracle_rag" if args.oracle else "vanilla_rag",
        }
        results.append(out)

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "vanilla_rag.jsonl")
    elif args.output:
        output_path = args.output
    else:
        output_path = "vanilla_rag_eval.jsonl"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved results to {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
