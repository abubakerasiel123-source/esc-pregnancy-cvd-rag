"""Evaluate the generation stage, not just retrieval.

biko.py --tune measures whether the right chunks are *found*. This measures
whether the answer built from them is *faithful*: every claim carries a
citation, every citation points at a source that was actually supplied, and
out-of-scope questions get refused rather than answered.

Run:  /opt/anaconda3/envs/rag/bin/python evaluate.py
Needs ANTHROPIC_API_KEY. Every query is a billed API call.
"""
import os
import re
import sys

import config  # noqa: F401 -- populates os.environ from .env on import
from biko import GOLD, PERSIST_DIR, RAGIndex
import generator

K = 5

# Questions the document cannot answer. The system should decline these, either
# via the retrieval cutoff or by the model saying the sources don't cover it.
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I train a convolutional neural network?",
    "What is the recommended dose of ibuprofen for a toddler's fever?",
    "What are the symptoms of appendicitis?",
]

CITATION = re.compile(r"\[(\d+)\]")
# A sentence making a clinical claim should cite. Sentences that are pure
# framing ("Here is what the guideline says:") legitimately do not.
FRAMING = re.compile(
    r"^(here|the (sources?|guideline|document)|based on|in summary|note that|"
    r"this (does not|doesn't)|i (could not|couldn't|cannot))",
    re.I,
)


def citation_report(answer: str, n_sources: int) -> dict:
    """Structural faithfulness checks that need no second model call."""
    cited = {int(m) for m in CITATION.findall(answer)}
    dangling = {c for c in cited if c < 1 or c > n_sources}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    substantive = [s for s in sentences if len(s) > 40 and not FRAMING.match(s)]
    uncited = [s for s in substantive if not CITATION.search(s)]

    return {
        "n_cited": len(cited),
        "dangling": sorted(dangling),
        "n_substantive": len(substantive),
        "n_uncited": len(uncited),
        "uncited_examples": uncited[:2],
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set -- generation cannot be evaluated.")

    index = RAGIndex(persist_directory=PERSIST_DIR)
    if not index.open_existing():
        sys.exit(f"No index at {PERSIST_DIR}. Run: python biko.py --build")

    print(f"Evaluating generation over {len(GOLD)} in-scope + "
          f"{len(OUT_OF_SCOPE)} out-of-scope questions (k={K})\n")

    total_dangling = total_uncited = total_substantive = 0
    cited_right_section = 0

    for query, accepted in GOLD:
        results = index.search(query, k=K)
        if generator.is_out_of_scope(results):
            print(f"  REFUSED (should not be): {query[:60]}")
            continue

        answer = generator.answer(query, results)
        rep = citation_report(answer, len(results))
        total_dangling += len(rep["dangling"])
        total_uncited += rep["n_uncited"]
        total_substantive += rep["n_substantive"]

        # Did the answer cite a chunk from a section the gold set accepts?
        cited_idx = {int(m) for m in CITATION.findall(answer)}
        if any(results[i - 1][0].metadata["section"].startswith(accepted)
               for i in cited_idx if 1 <= i <= len(results)):
            cited_right_section += 1

        flag = "!" if (rep["dangling"] or rep["n_uncited"]) else " "
        print(f"{flag} {query[:58]:58} cites={rep['n_cited']} "
              f"dangling={len(rep['dangling'])} uncited={rep['n_uncited']}"
              f"/{rep['n_substantive']}")
        for ex in rep["uncited_examples"]:
            print(f"      uncited: {ex[:88]}")

    refused = sum(
        1 for q in OUT_OF_SCOPE
        if generator.is_out_of_scope(index.search(q, k=K))
    )

    n = len(GOLD)
    print(f"\n{'-'*74}")
    print(f"cited an accepted section : {cited_right_section}/{n} "
          f"({cited_right_section/n:.0%})")
    print(f"dangling citations        : {total_dangling} (target 0)")
    print(f"uncited claim sentences   : {total_uncited}/{total_substantive}")
    print(f"out-of-scope refused      : {refused}/{len(OUT_OF_SCOPE)}")
    print(f"{'-'*74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
