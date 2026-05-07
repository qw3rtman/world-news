"""
Evaluate the agentic RAG pipeline on all question types.

Runs plan → retrieve → extract → answer on all eval_sample/ question types,
then scores with a 3-way LLM judge (MATCH / NONMATCH / REFUSAL).

Usage:
    cd rag-benchmarks/agentic
    python evaluate.py --config ../config.yaml
"""

import argparse
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import yaml
from vllm import SamplingParams

from retrieve import HybridRetriever
from generate import Generator
from agents import Agents
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


def load_eval_questions(eval_dir: str, types_filter: list[str] = None) -> list[dict]:
    """Load question types from eval_sample/, normalizing answer fields.

    If types_filter is provided, only load those types (e.g. ['direct', 'temporal']).
    """
    questions = []
    for filename in TASK_FILES:
        if types_filter:
            task_name = filename.replace(".jsonl", "")
            if task_name not in types_filter:
                continue
        path = os.path.join(eval_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue

        with open(path) as f:
            for line in f:
                q = json.loads(line)
                task = q.get("task", filename.replace(".jsonl", ""))
                q["split"] = task

                if "answer_after_update" in q:
                    q["answer"] = q["answer_after_update"]
                elif "gold_answer" in q:
                    q["answer"] = q["gold_answer"]

                q["expected_answer"] = q.get("answer", "")
                questions.append(q)

        count = sum(1 for qq in questions if qq["split"] == task)
        print(f"  {filename}: {count} questions")

    return questions


def parse_verdict(text: str) -> str:
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


def run_judge_api(results: list[dict], model_spec: str,
                  max_workers: int = 50) -> list[dict]:
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

    total = len(results)
    errors = [0]

    def _judge_one(idx):
        r = results[idx]
        prompt = JUDGE_PROMPT.format(
            question=r["question"],
            expected_answer=r.get("expected_answer", r.get("answer", "")),
            model_answer=r["model_answer"],
        )
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=128,
            )
            r["judge_verdict"] = parse_verdict(response.choices[0].message.content or "")
        except Exception as e:
            errors[0] += 1
            print(f"  Judge error on idx {idx}: {e}")
            r["judge_verdict"] = "nonmatch"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_judge_one, i) for i in range(total)]
        done = 0
        for f in futures:
            f.result()
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  Judge progress: {done}/{total} errors={errors[0]}")

    return results


def run_judge_vllm(llm, tokenizer, results: list[dict]) -> list[dict]:
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
    splits = sorted(set(r["split"] for r in results))
    total_n = len(results)
    total_match = sum(1 for r in results if r["judge_verdict"] == "match")

    for split in splits:
        sr = [r for r in results if r["split"] == split]
        n = len(sr)
        match = sum(1 for r in sr if r["judge_verdict"] == "match")
        nonmatch = sum(1 for r in sr if r["judge_verdict"] == "nonmatch")
        refusal = sum(1 for r in sr if r["judge_verdict"] == "refusal")
        print(f"\n{split} (N={n}):")
        print(f"  MATCH:    {match:4d} ({100*match/n:5.1f}%)")
        print(f"  NONMATCH: {nonmatch:4d} ({100*nonmatch/n:5.1f}%)")
        print(f"  REFUSAL:  {refusal:4d} ({100*refusal/n:5.1f}%)")

    if total_n:
        print(f"\nOverall (N={total_n}): MATCH={total_match}/{total_n} "
              f"({100*total_match/total_n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="openai:gpt-5.4-mini",
                        help="Judge model: 'openai:MODEL' for API, or local path for vLLM")
    parser.add_argument("--types", nargs="*", default=None,
                        help="Question types to evaluate (e.g. direct temporal)")
    parser.add_argument("--micro-batch-size", type=int, default=0,
                        help="Micro-batch size for pipelined retrieve+extract (0=disabled)")
    parser.add_argument("--reranker-devices", nargs="*", default=["cuda:2"],
                        help="GPU devices for reranker (e.g. cuda:2 cuda:3)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override reranking top_n from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["generation"]["model"]
    if model_cfg.startswith("${") and model_cfg.endswith("}"):
        cfg["generation"]["model"] = os.environ[model_cfg[2:-1]]

    artifacts_dir = cfg["artifacts_dir"]
    eval_dir = cfg["eval_dir"]

    # Load question types (optionally filtered)
    print("Loading evaluation questions...")
    all_questions = load_eval_questions(eval_dir, types_filter=args.types)
    print(f"Loaded {len(all_questions)} total questions")

    # Initialize components
    print("Loading retriever...")
    retriever = HybridRetriever(
        corpus_path=os.path.join(artifacts_dir, "corpus", "corpus.jsonl"),
        embeddings_path=os.path.join(artifacts_dir, "embeddings.npy"),
        embedding_model=cfg["embeddings"]["model"],
        index_dir=os.path.join(artifacts_dir, "index"),
        bm25_top_k=cfg["retrieval"]["bm25_top_k"],
        dense_top_k=cfg["retrieval"]["dense_top_k"],
        reranker_model=cfg["reranking"]["model"],
        top_n=args.top_n or cfg["reranking"]["top_n"],
        reranker_devices=args.reranker_devices,
    )

    print("Loading generator...")
    gen_cfg = cfg["generation"]
    generator = Generator(
        model=gen_cfg["model"],
        tp=gen_cfg["tp"],
        max_model_len=gen_cfg["max_model_len"],
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg.get("top_p", 0.8),
        top_k=gen_cfg.get("top_k", 20),
        min_p=gen_cfg.get("min_p", 0),
    )

    agents = Agents(
        tokenizer=generator.tokenizer,
        max_steps=cfg["agent"]["max_steps"],
    )

    # Run pipeline
    results = run_pipeline(retriever, generator, agents, all_questions,
                           micro_batch_size=args.micro_batch_size)

    # Free GPU memory before judge
    del generator, retriever, agents
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Carry split tags
    for r, q in zip(results, all_questions):
        r["split"] = q["split"]
        r["condition"] = "agentic_rag"

    # LLM judge
    print(f"\nRunning LLM judge ({args.judge_model}) on {len(results)} results...")
    if ":" in args.judge_model:
        results = run_judge_api(results, args.judge_model)
    else:
        # Can't reuse generator (freed above) — would need separate load
        raise RuntimeError("Local vLLM judge requires generator to still be loaded. "
                           "Use API judge (e.g. openai:gpt-5.4-mini) with agentic eval.")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "agentic_rag.jsonl")
    elif args.output:
        output_path = args.output
    else:
        output_path = "agentic_rag_eval.jsonl"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved results to {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
