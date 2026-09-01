"""Grounded answer generation over chunks retrieved by biko.py.

This is the "G" of RAG. The contract is deliberately strict: the model answers
only from the numbered sources it is handed, cites them inline, and says so
plainly when the sources do not cover the question. The source document is a
clinical guideline, so an invented answer is worse than no answer.
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

import anthropic
from langchain_core.documents import Document

MODEL = "claude-opus-5"
MAX_TOKENS = 8000

# Above this cosine distance, retrieval is treated as "nothing relevant found"
# and no request is sent. Measured with calibrate_threshold() on bge-small:
# in-scope questions top out at 0.225, out-of-scope ones start at 0.411, so the
# midpoint separates them cleanly. Re-run that helper if the embedding model
# changes -- distances are not comparable across models.
RELEVANCE_CUTOFF = 0.32

SYSTEM_PROMPT = """\
You answer questions about one specific document: the 2025 ESC Guidelines for \
the management of cardiovascular disease and pregnancy. The user is shown the \
same numbered sources you are.

Rules, in order of importance:

1. Answer ONLY from the numbered sources below. You have no other knowledge of \
this document. Do not fill gaps from general medical training.
2. Cite the source number inline for every clinical claim, like [1] or [2][3]. \
A sentence carrying a clinical claim with no citation is a failure.
3. If the sources do not answer the question, say exactly what is missing \
rather than guessing. Partial answers are fine if you mark what is missing.
4. If sources disagree or are ambiguous, say so instead of picking one silently.
5. Preserve clinical qualifiers exactly: recommendation classes (I, IIa, IIb, \
III), levels of evidence (A, B, C), doses, and gestational timing. Never round \
a dose or drop a qualifier.
6. Be concise. Lead with the direct answer, then the supporting detail.

This is a document-lookup tool for clinicians and students. It reports what the \
guideline says; it is not medical advice and must never be phrased as advice \
for a specific patient. Do not add a safety disclaimer -- the interface \
already carries one."""


def format_sources(results: List[Tuple[Document, float]]) -> str:
    """Render retrieved chunks as the numbered source block the prompt cites."""
    blocks = []
    for i, (doc, _distance) in enumerate(results, 1):
        m = doc.metadata
        pages = (str(m["page"]) if m["page"] == m["page_end"]
                 else f"{m['page']}-{m['page_end']}")
        blocks.append(
            f"[{i}] Section: {m['section']} | Page: {pages}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def is_out_of_scope(results: List[Tuple[Document, float]]) -> bool:
    """True when even the best chunk is too far to be worth answering from."""
    return not results or results[0][1] > RELEVANCE_CUTOFF


def build_messages(
    query: str,
    results: List[Tuple[Document, float]],
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Prior turns as plain text, then this turn's sources plus the question."""
    messages: List[Dict[str, Any]] = list(history or [])
    messages.append({
        "role": "user",
        "content": (
            f"Sources:\n\n{format_sources(results)}\n\n"
            f"---\n\nQuestion: {query}"
        ),
    })
    return messages


def answer_stream(
    query: str,
    results: List[Tuple[Document, float]],
    history: Optional[List[Dict[str, Any]]] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> Iterator[str]:
    """Stream a grounded answer.

    Streaming matters twice over: it keeps the UI responsive, and adaptive
    thinking makes single responses long enough that a non-streaming call can
    reach the SDK's HTTP timeout.
    """
    client = client or anthropic.Anthropic()

    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        # Auto-caches the last cacheable block. Worth little on a one-shot
        # question (the source block differs every time), but it pays off as
        # chat history accumulates across turns.
        cache_control={"type": "ephemeral"},
        thinking={"type": "adaptive"},
        messages=build_messages(query, results, history),
    ) as stream:
        for text in stream.text_stream:
            yield text


def answer(
    query: str,
    results: List[Tuple[Document, float]],
    history: Optional[List[Dict[str, Any]]] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    """Non-streaming convenience wrapper, used by the evaluation harness."""
    return "".join(answer_stream(query, results, history, client))


def calibrate_threshold(index, in_scope: List[str], out_of_scope: List[str]) -> None:
    """Print top-1 distances for in- and out-of-scope queries.

    RELEVANCE_CUTOFF should sit in the gap between the two clusters. Re-run this
    after changing the embedding model -- distances are not comparable across
    models.
    """
    def top1(q: str) -> float:
        r = index.search(q, k=1)
        return r[0][1] if r else float("inf")

    ins = sorted(top1(q) for q in in_scope)
    outs = sorted(top1(q) for q in out_of_scope)
    print(f"in-scope   top-1 distance: min={ins[0]:.3f} max={ins[-1]:.3f}")
    print(f"out-scope  top-1 distance: min={outs[0]:.3f} max={outs[-1]:.3f}")
    if ins[-1] < outs[0]:
        print(f"clean separation -- put the cutoff between "
              f"{ins[-1]:.3f} and {outs[0]:.3f}")
    else:
        print("OVERLAP -- no threshold separates these cleanly")
