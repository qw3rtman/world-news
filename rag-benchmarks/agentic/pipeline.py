"""
End-to-end agentic RAG pipeline: plan → retrieve → extract → answer.

Supports micro-batched pipeline parallelism: while the LLM GPUs extract notes
for micro-batch N, the reranker GPU(s) retrieve for micro-batch N+1.

Usage:
    python pipeline.py --config ../config.yaml --questions ../../eval_sample/all.jsonl
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import yaml

from retrieve import HybridRetriever
from generate import Generator
from agents import Agents


def load_questions(path: str) -> list[dict]:
    """Load questions from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_pipeline(retriever: HybridRetriever, generator: Generator,
                 agents: Agents, questions: list[dict],
                 micro_batch_size: int = 0) -> list[dict]:
    """Run the full agentic pipeline with optional micro-batch pipelining."""
    n = len(questions)
    use_pipeline = micro_batch_size > 0

    # Phase 1: Plan
    print(f"Planning for {n} questions...")
    plan_prompts = [agents.format_plan_prompt(q["question"]) for q in questions]
    plan_results = generator.generate_batch(plan_prompts, max_tokens=512)
    plans = [agents.parse_plan(r["text"]) for r in plan_results]
    plan_token_counts = [{"prompt_tokens": r["prompt_tokens"],
                          "completion_tokens": r["completion_tokens"]}
                         for r in plan_results]

    step_counts = [len(p) for p in plans]
    print(f"Plan steps — min: {min(step_counts)}, max: {max(step_counts)}, "
          f"avg: {sum(step_counts) / n:.1f}")

    # Phase 2: Execute
    all_notes: list[list[str]] = [[] for _ in questions]
    traces: list[dict] = [{"plan": plans[i], "plan_tokens": plan_token_counts[i],
                            "steps": []} for i in range(n)]
    max_active_steps = max(step_counts)

    for step_idx in range(max_active_steps):
        active_idx = [i for i in range(n) if step_idx < len(plans[i])]

        print(f"Step {step_idx + 1}/{max_active_steps}: "
              f"defining tasks for {len(active_idx)} questions...")
        step_definer_prompts = [
            agents.format_step_definer_prompt(
                plan=plans[i],
                cur_step=plans[i][step_idx],
                notes_so_far=all_notes[i],
            )
            for i in active_idx
        ]
        definer_results = generator.generate_batch(step_definer_prompts)
        step_tasks = [agents.parse_step_task(r["text"]) for r in definer_results]

        for j, task in enumerate(step_tasks):
            traces[active_idx[j]]["steps"].append({
                "step_idx": step_idx,
                "description": plans[active_idx[j]][step_idx],
                "type": task["type"],
                "query": task["task"],
                "chunks": [],
                "notes": "",
                "define_tokens": {"prompt_tokens": definer_results[j]["prompt_tokens"],
                                  "completion_tokens": definer_results[j]["completion_tokens"]},
            })

        search_local = [j for j, t in enumerate(step_tasks) if t["type"] == "search"]
        aggregate_local = [j for j, t in enumerate(step_tasks) if t["type"] == "aggregate"]

        if search_local:
            search_queries = [step_tasks[j]["task"] for j in search_local]
            search_global = [active_idx[j] for j in search_local]

            if use_pipeline and len(search_local) > micro_batch_size:
                _run_search_pipelined(
                    retriever, generator, agents,
                    search_queries, search_global,
                    all_notes, traces, step_idx, max_active_steps,
                    micro_batch_size,
                )
            else:
                _run_search_batch(
                    retriever, generator, agents,
                    search_queries, search_global,
                    all_notes, traces, step_idx, max_active_steps,
                )

        if aggregate_local:
            aggregate_tasks = [step_tasks[j]["task"] for j in aggregate_local]
            aggregate_global = [active_idx[j] for j in aggregate_local]

            print(f"Step {step_idx + 1}/{max_active_steps}: "
                  f"aggregating for {len(aggregate_global)} aggregate tasks...")
            aggregate_prompts = [
                agents.format_aggregate_prompt(task) for task in aggregate_tasks
            ]
            aggregate_results = generator.generate_batch(aggregate_prompts)

            for k, i in enumerate(aggregate_global):
                all_notes[i].append(agents.parse_notes(aggregate_results[k]["text"]))
                traces[i]["steps"][-1]["notes"] = all_notes[i][-1]
                traces[i]["steps"][-1]["aggregate_tokens"] = {
                    "prompt_tokens": aggregate_results[k]["prompt_tokens"],
                    "completion_tokens": aggregate_results[k]["completion_tokens"],
                }

    # Phase 3: Answer (QA style)
    print(f"Generating final answers for {n} questions...")
    answer_prompts = [
        agents.format_answer_prompt(q["question"], all_notes[i])
        for i, q in enumerate(questions)
    ]
    answer_results = generator.generate_batch(answer_prompts)

    results = []
    for i, (q, ar) in enumerate(zip(questions, answer_results)):
        traces[i]["answer_tokens"] = {
            "prompt_tokens": ar["prompt_tokens"],
            "completion_tokens": ar["completion_tokens"],
        }
        results.append({
            "question": q["question"],
            "expected_answer": q.get("answer", q.get("expected_answer", "")),
            "model_answer": ar["text"],
            "num_steps": len(plans[i]),
            "trace": traces[i],
        })

    return results


def _run_search_batch(retriever, generator, agents,
                      search_queries, search_global,
                      all_notes, traces, step_idx, max_active_steps):
    """Standard: retrieve all, then extract all."""
    print(f"Step {step_idx + 1}/{max_active_steps}: "
          f"retrieving for {len(search_global)} search tasks...")
    chunks_list = retriever.retrieve_batch(search_queries)

    extract_prompts = [
        agents.format_extract_prompt(search_queries[k], chunks_list[k])
        for k in range(len(search_queries))
    ]
    extract_results = generator.generate_batch(extract_prompts, max_tokens=512)

    for k, i in enumerate(search_global):
        all_notes[i].append(agents.parse_notes(extract_results[k]["text"]))
        traces[i]["steps"][-1]["chunks"] = chunks_list[k]
        traces[i]["steps"][-1]["notes"] = all_notes[i][-1]
        traces[i]["steps"][-1]["extract_tokens"] = {
            "prompt_tokens": extract_results[k]["prompt_tokens"],
            "completion_tokens": extract_results[k]["completion_tokens"],
        }


def _run_search_pipelined(retriever, generator, agents,
                          search_queries, search_global,
                          all_notes, traces, step_idx, max_active_steps,
                          micro_batch_size):
    """Pipelined: overlap retrieval of mb_{i+1} with extraction of mb_i."""
    mbs = micro_batch_size
    n_queries = len(search_queries)
    n_mbs = (n_queries + mbs - 1) // mbs

    print(f"Step {step_idx + 1}/{max_active_steps}: "
          f"pipelined retrieve+extract for {n_queries} search tasks "
          f"({n_mbs} micro-batches of {mbs})...")

    # Retrieve first micro-batch
    mb0_end = min(mbs, n_queries)
    cur_chunks = retriever.retrieve_batch(search_queries[:mb0_end])

    all_chunks = []
    all_extract_results = []

    for mb_idx in range(n_mbs):
        mb_start = mb_idx * mbs
        mb_end = min(mb_start + mbs, n_queries)
        mb_queries = search_queries[mb_start:mb_end]

        extract_prompts = [
            agents.format_extract_prompt(mb_queries[k], cur_chunks[k])
            for k in range(len(mb_queries))
        ]

        if mb_idx < n_mbs - 1:
            next_start = (mb_idx + 1) * mbs
            next_end = min(next_start + mbs, n_queries)
            next_queries = search_queries[next_start:next_end]

            with ThreadPoolExecutor(max_workers=2) as pool:
                extract_future = pool.submit(
                    generator.generate_batch, extract_prompts, 512)
                retrieve_future = pool.submit(
                    retriever.retrieve_batch, next_queries)
                mb_extract_results = extract_future.result()
                next_chunks = retrieve_future.result()
        else:
            mb_extract_results = generator.generate_batch(
                extract_prompts, max_tokens=512)
            next_chunks = None

        all_chunks.extend(cur_chunks)
        all_extract_results.extend(mb_extract_results)
        cur_chunks = next_chunks

    for k, i in enumerate(search_global):
        all_notes[i].append(agents.parse_notes(all_extract_results[k]["text"]))
        traces[i]["steps"][-1]["chunks"] = all_chunks[k]
        traces[i]["steps"][-1]["notes"] = all_notes[i][-1]
        traces[i]["steps"][-1]["extract_tokens"] = {
            "prompt_tokens": all_extract_results[k]["prompt_tokens"],
            "completion_tokens": all_extract_results[k]["completion_tokens"],
        }


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
        corpus_path=os.path.join(artifacts_dir, "corpus", "corpus.jsonl"),
        embeddings_path=os.path.join(artifacts_dir, "embeddings.npy"),
        embedding_model=cfg["embeddings"]["model"],
        index_dir=os.path.join(artifacts_dir, "index"),
        bm25_top_k=cfg["retrieval"]["bm25_top_k"],
        dense_top_k=cfg["retrieval"]["dense_top_k"],
        reranker_model=cfg["reranking"]["model"],
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

    agents = Agents(
        tokenizer=generator.tokenizer,
        max_steps=cfg["agent"]["max_steps"],
    )

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    results = run_pipeline(retriever, generator, agents, questions)

    output_path = args.output or args.questions.replace(".jsonl", "_agentic_results.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
