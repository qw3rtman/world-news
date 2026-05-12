"""
Agent prompt templates and parsers for the agentic RAG pipeline.

Four agents:
  - Planner:      decomposes a question into high-level step descriptions
  - Step Definer: converts each step into a specific retrieval query or
                  aggregate task, conditioned on accumulated notes so far
  - Extractor:    pulls relevant sentences from retrieved documents for a sub-query
  - QA:           synthesizes a final answer from all accumulated notes

Used as a module by pipeline.py, not run directly.
"""

import json


PLANNER_SYSTEM = """\
You are tasked with assisting users in generating structured plans for answering \
questions. Your goal is to deconstruct a query into manageable, simpler components. \
The current date is April 1, 2026.

For each question, perform the following tasks:

*Analysis: Identify the core components of the question, emphasizing the key \
elements and context needed for a comprehensive understanding. Determine whether \
the question is straightforward or requires multiple steps to provide an accurate answer.

*Plan Creation:
- Break down the question into smaller, simpler questions by reasoning that lead \
to the final answer. Ensure those steps are non overlap. Stop at the step where \
its answer can be the final answer.
- Ensure each step is clear and logically sequenced.
- Consider any past attempts or experiences provided as context, and use them to \
refine or adjust the plan to avoid past pitfalls.
- Each step is a question to search, or to aggregate output from previous steps. \
Do not verify previous step.
- Your task is planning, not answering. Do not put any answer from your knowledge \
into the plan.

# Notes:
- Your task is to provide clarity and guidance on the approach to answering, \
rather than providing the final answer directly.
- Put your output in a list of string, each string describe a sub-task

# Example plan:
Question: What country of origin does House of Cosbys and Bill Cosby have in common?
Steps: ["Determine the country of origin for House of Cosbys.", "Determine the country of origin for Bill Cosby.", "From previous answers, which is the common country"]

Question: Which film has the director who died later, The House Of Tears or College Ranga?
Steps: ["Identify the director of The House Of Tears", "Identify the director of College Ranga", "When did the director of The House Of Tears die", "When did the director of College Ranga die", "Compare the death dates of the two directors to determine which one died later."]

Question: Peter Griffith's granddaughter had her screen debut in what 1999 film?
Steps: ["Who is Peter Griffith's granddaughter", "What 1999 film did she have screen debut"]

Question: how many episodes are in chicago fire season 4?
Steps: ["how many episodes are in chicago fire season 4"]

Question: Are both directors of films The Stoneman Murders and Chandralekha (2014 Film) from the same country?
Steps: ["Who is the director of film The Stoneman Murders", "Who is the director of film Chandralekha (2014 Film)", "Determine the country of origin for the director of The Stoneman Murders", "Determine the country of origin for the director of Chandralekha (2014 Film)", "Compare the two countries to determine if they are the same"]"""

PLANNER_USER = """\
Question: {question}
Past experience:
{memory}"""


STEP_DEFINER_SYSTEM = """\
Given a plan, the current step, and the results from finished steps, decide the \
task for this step. The current date is April 1, 2026.
Output the type of task and the query.

Task types:
- "search": a specific query to search a document corpus for relevant passages
- "aggregate": a self-contained reasoning task (comparison, synthesis, counting) \
that can be answered from the previous step results without additional retrieval

The query need to be in detail (do not put "based on the previous results" in the query)
Include all of information from previous step's results in the query if it maked, \
especially for aggregate task
Be concise.

Output JSON only: {{"type": "search", "task": "..."}}"""

STEP_DEFINER_USER = """\
Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}"""


EXTRACTOR_SYSTEM = """\
Summarize and extract all relevant information from the provided passages based \
on the given question. Remove all irrelevant information but treat all \
information provided as factual. Think step-by-step.
The current date is April 1, 2026.

# Steps
1. **Identify Key Elements**: Read the question carefully to determine what \
specific information is being requested.
2. **Analyze Passages**: Review the passages thoroughly to find any segments \
that contain information relevant to the question.
3. **Extract Relevant Information**: Highlight or note down sentences, phrases, \
or words from the passages that relate to the question.
4. **Remove Irrelevant Details**: Ensure that all extracted information is \
relevant to the question, eliminating any unnecessary or unrelated content.

# Output Format
- Output a list of notes. Each note contains related information from the \
passage as well as precise evidences and why.
- Each note is clear, standalone.

# Notes
- Avoiding any irrelevant details.
- If a piece of information is mentioned in multiple places, include it only once.
- If there are no related information, output: No related information from this document."""

EXTRACTOR_USER = """\
Passage:
###
{passages}
###

Query: {query}"""


QA_SYSTEM = """\
You are an assistant for question-answering tasks. The current date is April 1, 2026. \
Use the following process to deliver concise and precise answers based on the \
retrieved context. If the notes do not contain enough information to answer, say so.

1. **Analyze Carefully**: Begin by thoroughly analyzing both the question and \
the provided context.
2. **Identify Core Details**: Focus on identifying the essential names, terms, \
or details that directly answer the question. Disregard any irrelevant information.
3. **Provide a Concise Answer**:
   - Remove redundant words and extraneous details.
   - Present the answer by listing only the necessary names, terms, or very \
brief facts that are crucial for answering the question.
4. **Clarity and Accuracy**: Ensure that your answer is clear and maintains \
the original meaning of the information provided.
5. **Consensus**: If the contexts are not consensus, pick one which is the \
most logical, consensus, or confident."""

QA_USER = """\
Retrieved documents:
{notes}
Question: {question}"""


AGGREGATE_SYSTEM = """\
Answer the question provided. The current date is April 1, 2026.
Provide a Concise Answer:
- Remove redundant words and extraneous details.
- Present the answer by listing only the necessary names, terms, or brief facts \
that are crucial for answering the question.
- If you have multiple answers, only output one answer which is most confident
- Treat information provided as context as factual
Think step-by-step"""

AGGREGATE_USER = """\
{task}"""


class Agents:
    """Prompt formatting and response parsing for all four agents."""

    def __init__(self, tokenizer, max_steps: int):
        self.tokenizer = tokenizer
        self.max_steps = max_steps

    def format_plan_prompt(self, question: str) -> str:
        user = PLANNER_USER.format(question=question, memory="")
        return self._apply_chat_with_system(PLANNER_SYSTEM, user)

    def parse_plan(self, text: str) -> list[str]:
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    steps = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
                    if steps:
                        return steps[: self.max_steps]
            except (json.JSONDecodeError, ValueError):
                pass

        steps = []
        for line in text.splitlines():
            line = line.strip().lstrip("0123456789.-) ").strip()
            if line:
                steps.append(line)

        return steps[: self.max_steps] if steps else [text]

    def format_step_definer_prompt(self, plan: list[str], cur_step: str,
                                   notes_so_far: list[str]) -> str:
        plan_str = str(plan)
        if notes_so_far:
            memory = "\n\n".join(
                f"Step {i + 1}: {plan[i]}\nNotes: {note}"
                for i, note in enumerate(notes_so_far)
            )
        else:
            memory = "No previous steps completed."
        user = STEP_DEFINER_USER.format(
            plan=plan_str, cur_step=cur_step, memory=memory,
        )
        return self._apply_chat_with_system(STEP_DEFINER_SYSTEM, user)

    def parse_step_task(self, text: str) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and "task" in parsed:
                    task_type = str(parsed.get("type", "search")).lower()
                    if task_type not in ("search", "aggregate"):
                        task_type = "search"
                    return {"type": task_type, "task": str(parsed["task"]).strip()}
            except (json.JSONDecodeError, ValueError):
                pass

        task_type = "aggregate" if "aggregate" in text.lower() else "search"
        return {"type": task_type, "task": text}

    def format_extract_prompt(self, sub_query: str, chunks: list[str]) -> str:
        passages = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
        user = EXTRACTOR_USER.format(query=sub_query, passages=passages)
        return self._apply_chat_with_system(EXTRACTOR_SYSTEM, user)

    def parse_notes(self, text: str) -> str:
        return text.strip()

    def format_aggregate_prompt(self, task: str) -> str:
        user = AGGREGATE_USER.format(task=task)
        return self._apply_chat_with_system(AGGREGATE_SYSTEM, user)

    def format_answer_prompt(self, question: str, notes: list[str]) -> str:
        notes_text = "\n\n".join(f"[Step {i + 1}] {note}" for i, note in enumerate(notes))
        user = QA_USER.format(question=question, notes=notes_text)
        return self._apply_chat_with_system(QA_SYSTEM, user)

    def _apply_chat_with_system(self, system: str, user: str) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
