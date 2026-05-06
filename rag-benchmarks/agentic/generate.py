"""
LLM generation via vLLM.

Wraps a single vLLM LLM instance shared across all agent calls (planner,
extractor, QA) and the judge in evaluate.py. Prompt formatting is handled
by agents.py; this module only handles inference.

Used as a module by pipeline.py, not run directly.
"""

from vllm import LLM, SamplingParams


class Generator:
    """Loads model via vLLM once, generates outputs for pre-built prompts."""

    def __init__(self, model: str, tp: int, max_model_len: int,
                 max_tokens: int, temperature: float,
                 top_p: float = 0.8, top_k: int = 20, min_p: float = 0):
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            trust_remote_code=True,
            additional_config={"gdn_prefill_backend": "triton"},
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

    def generate_batch(self, prompts: list[str], max_tokens: int = None) -> list[dict]:
        """Generate outputs for a batch of pre-formatted prompts.

        Returns list of dicts with keys: text, prompt_tokens, completion_tokens.

        Processes in sub-batches of 200 to avoid vLLM V1 engine SIGSEGV
        on large batch completion cleanup.
        """
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=max_tokens or self.max_tokens,
        )
        BATCH_SIZE = 200
        all_results = []
        for start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[start:start + BATCH_SIZE]
            outputs = self.llm.generate(batch, params)
            all_results.extend([{
                "text": o.outputs[0].text.strip(),
                "prompt_tokens": len(o.prompt_token_ids),
                "completion_tokens": len(o.outputs[0].token_ids),
            } for o in outputs])
        return all_results
