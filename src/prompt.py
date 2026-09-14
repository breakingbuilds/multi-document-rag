"""
src/prompt.py
-------------
Every prompt template used in the project lives here, in one place, so the
wording can be tuned without touching pipeline code.

Groups of prompts
    1. ANSWERING  - build_rag_messages(): the grounded question-answering
                    prompt with numbered context passages and [n] citations.
    2. REWRITING  - templates for the LLM-based query rewriter
                    (multi-query paraphrases, HyDE, step-back).
    3. JUDGING    - templates for the manual evaluation metrics
                    (claim extraction, claim verification, relevance rating).

Design principles for the answering prompt
    * The context passages are NUMBERED and each carries its source locator
      ("[2] employee_handbook.docx - section: Leave at a Glance"). The model
      is told to cite those numbers, which is what lets the pipeline turn "[2]"
      into a clickable source card and lets the reader verify the answer.
    * The model is explicitly allowed -- required -- to say it does not know.
      That single instruction is the biggest lever against hallucination and
      is what the out-of-scope question in the evaluation set tests.
"""

from loaders.base import describe_location

# ---------------------------------------------------------------------- #
# 1. Answering
# ---------------------------------------------------------------------- #
RAG_SYSTEM_PROMPT = """You are a careful assistant that answers questions about a company using ONLY the context passages provided by the user.

Rules you must follow:
1. Use only facts that appear in the context passages. Never invent numbers, dates, names or policies.
2. After every fact you state, cite the passage number(s) it came from in square brackets, e.g. "Employees receive 20 annual leave days [1][3]."
3. If the passages do not contain the information needed, reply with exactly this sentence first:
   "I don't have enough information in the provided documents to answer that."
   You may then briefly mention any related information that IS in the passages.
4. If passages disagree, say so and cite both.
5. Be concise: give the direct answer in the first sentence, then supporting details. Do not repeat the question."""

NO_ANSWER_SENTENCE = "I don't have enough information in the provided documents to answer that."


def format_context(chunks) -> str:
    """Render retrieved chunks as numbered, source-labelled passages.

    `chunks` is any sequence of objects exposing `.text` and `.metadata`
    (RetrievedChunk from src/retriever.py, or SearchHit).
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        locator = describe_location(chunk.metadata)
        label = f"{source} - {locator}" if locator else source
        blocks.append(f"[{index}] ({label})\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, chunks) -> list[dict]:
    """Return the chat messages for the grounded-answer call."""
    context = format_context(chunks) if chunks else "(no passages were retrieved)"
    user_prompt = (
        "Context passages:\n"
        f"{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer (with [n] citations):"
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------- #
# 2. Query rewriting (LLM-based)
# ---------------------------------------------------------------------- #
# Shared rules, mirrored from the manual rewriter: rewriting must never
# answer the question and must never add facts the user did not state.
REWRITE_RULES = """Rules:
- Do NOT answer the question. Output search queries only.
- Preserve the original meaning exactly; do not add facts, numbers or assumptions.
- Use vocabulary likely to appear in company policy documents, handbooks, FAQs and HR records.
- No numbering, no bullets, no commentary."""

MULTI_QUERY_SYSTEM = f"""You rewrite user questions into alternative search queries for a document retrieval system.
{REWRITE_RULES}
Return exactly {{n}} alternative queries, one per line."""

MULTI_QUERY_USER = "Original question: {question}"

HYDE_SYSTEM = """You write a short hypothetical passage that could appear in a company's internal documents (policy handbook, HR FAQ, meeting notes) and that would directly answer the user's question.
This passage is used ONLY to search for real documents, so it does not need to be true -- but it must be written in the style and vocabulary of such documents.
Write 2-4 sentences. Do not mention that it is hypothetical. Do not add headings or commentary."""

HYDE_USER = "Question: {question}\n\nHypothetical passage:"

STEP_BACK_SYSTEM = f"""You turn a specific question into ONE broader, more general question about the same underlying topic, so that background material can be retrieved.
Example: "Can I carry over 7 unused leave days?" -> "What are the rules for annual leave carryover?"
{REWRITE_RULES}
Return only the single broader question."""

STEP_BACK_USER = "Specific question: {question}"


# ---------------------------------------------------------------------- #
# 3. Judging (manual evaluation metrics)
# ---------------------------------------------------------------------- #
# These are simplified versions of the prompts RAGAS uses internally, so the
# manual metrics are directly comparable with the RAGAS ones. All three ask
# for JSON so the scores can be parsed reliably.

CLAIM_EXTRACTION_SYSTEM = """You break an answer into its atomic factual claims.
A claim is a single, self-contained statement that can be checked as true or false.
Ignore hedges, apologies and citation markers such as [1].
If the answer says it does not have enough information, return an empty list.
Return JSON only: {"claims": ["...", "..."]}"""

CLAIM_EXTRACTION_USER = "Question: {question}\n\nAnswer: {answer}\n\nJSON:"

CLAIM_VERIFICATION_SYSTEM = """You are a strict fact checker. For each claim, decide whether it is directly supported by the context passages.
A claim is "supported" only if the passages state it or it follows unambiguously from them. Otherwise it is not supported.
Return JSON only: {"verdicts": [{"claim": "...", "supported": true, "reason": "..."}, ...]}
Keep the claims in the same order as given."""

CLAIM_VERIFICATION_USER = "Context passages:\n{context}\n\nClaims:\n{claims}\n\nJSON:"

RELEVANCE_RATING_SYSTEM = """You rate how well an answer addresses a question, on a 1-5 scale:
5 = fully answers the question, focused, no padding
4 = answers it with minor gaps or extra material
3 = partially answers it
2 = mostly misses the point or is very incomplete
1 = does not address the question at all
An honest "I don't have enough information" reply to an unanswerable question counts as 5.
Return JSON only: {"score": <1-5>, "reason": "..."}"""

RELEVANCE_RATING_USER = "Question: {question}\n\nAnswer: {answer}\n\nJSON:"
