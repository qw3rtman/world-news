"""
LLM generation: build prompt from retrieved context and generate answer via vLLM.

Formats the reranked documents as context passages, appends the question,
and calls the generator model.

Used as a module by pipeline.py, not run directly.
"""

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """\
Read the following context passages, then answer the question.
The current date is April 1, 2026.

Context:
{context}

Question: {question}

Answer concisely and factually. If the context does not contain enough information, say "I don't know."
"""


class Generator:
    """Loads model via vLLM once, generates answers per query."""

    def __init__(self, model: str, tp: int, max_model_len: int,
                 max_tokens: int, temperature: float,
                 top_p: float = 0.8, top_k: int = 20, min_p: float = 0):
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            trust_remote_code=True,
            additional_config={"gdn_prefill_backend": "triton"},
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
        )

    def generate(self, question: str, docs: list[dict]) -> str:
        """Build prompt from documents and generate an answer."""
        prompt = self._build_prompt(question, docs)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(self, questions: list[str],
                       docs_per_question: list[list[dict]]) -> list[str]:
        """Generate answers for multiple questions at once."""
        prompts = [
            self._build_prompt(q, d)
            for q, d in zip(questions, docs_per_question)
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def _build_prompt(self, question: str, docs: list[dict]) -> str:
        """Format documents + question into a chat prompt."""
        context = "\n\n".join(self._format_doc(doc) for doc in docs)
        user_content = PROMPT_TEMPLATE.format(context=context, question=question)
        messages = [{"role": "user", "content": user_content}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    @staticmethod
    def _format_doc(doc: dict) -> str:
        """Format a document with its publish date header for the LLM context."""
        date = doc.get("publish_date", "")
        if date:
            date = date.split("T")[0]
            return f"[Published: {date}]\n{doc['contents']}"
        return doc["contents"]
