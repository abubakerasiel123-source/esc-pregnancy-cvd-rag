"""
Section-aware RAG pipeline over a PDF.

The deps live in the `rag` conda env, not in the system python3, so this file
re-launches itself into an interpreter that can import them. That means
`python3 biko.py`, `python biko.py` and the editor Run button all work.
"""
import os
import sys
import subprocess

# --- interpreter bootstrap (must run before any third-party import) ---------
_INTERPRETERS = (
    "/opt/anaconda3/envs/rag/bin/python",
    "/opt/anaconda3/bin/python",
)


_REQUIRED = ("pymupdf", "langchain_core", "langchain_text_splitters",
             "langchain_huggingface", "langchain_chroma", "chromadb",
             "sentence_transformers")
_PROBE = "import " + ", ".join(_REQUIRED)


def _relaunch_if_deps_missing() -> None:
    import importlib.util
    missing = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
    if not missing:
        return

    here = os.path.realpath(sys.executable)
    for candidate in _INTERPRETERS:
        if not os.path.exists(candidate) or os.path.realpath(candidate) == here:
            continue
        probe = subprocess.run([candidate, "-c", _PROBE], capture_output=True)
        if probe.returncode == 0:
            print(f"[bootstrap] {sys.executable} is missing {', '.join(missing)}; "
                  f"re-launching with {candidate}\n", file=sys.stderr)
            os.execv(candidate, [candidate, os.path.abspath(__file__), *sys.argv[1:]])

    sys.exit(
        f"Missing modules: {', '.join(missing)}, and no interpreter on this "
        "machine has the full set.\n"
        "Install them into the env you want to use, e.g.:\n"
        "  conda activate rag\n"
        "  pip install langchain-core langchain-text-splitters "
        "langchain-huggingface langchain-chroma "
        "sentence-transformers pymupdf"
    )


_relaunch_if_deps_missing()
# ---------------------------------------------------------------------------

import re
import bisect
from typing import List, Dict, Any, Tuple

import fitz # type: ignore

from langchain_core.documents import Document # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore

# Anchored to this file rather than to an absolute path or the working
# directory: biko.py is launched from the editor, from streamlit and from an
# arbitrary cwd, and a hardcoded path breaks the moment the project folder is
# moved or renamed -- which is exactly what happened when these files were
# collected into "RAG System/".
_HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(_HERE, "pregnancy_cvd_2025.pdf")
PERSIST_DIR = os.path.join(_HERE, "chroma_pdf_rag")

# ==========================================
# 1. PARSE PDF (Structured Extraction)
# ==========================================

# A few pages embed subset fonts with no usable ToUnicode map, so get_text()
# returns control characters there instead of letters. Those blocks are
# unreadable and must not reach the index.
_CONTROL = set(range(0, 32)) - {9, 10, 13} | set(range(0x7F, 0xA0))
_GARBAGE_THRESHOLD = 0.10

_MARGIN = 45.0             # pts: running header / footer band
_MIN_BLOCK_WIDTH = 20.0    # pts: anything narrower is rotated marginalia
_FULL_WIDTH_FRAC = 0.55    # a block this wide spans both columns
_HEADING_MAX_CHARS = 160   # headings are short; body paragraphs are not

_NUM_PREFIX = re.compile(r"^\d+(?:\.\d+)*\.?\s+")

# Back matter is bibliography and administrivia -- no clinical content, and it
# competes with real answers at retrieval time.
_BACK_MATTER = re.compile(
    r"^(references|appendix|author information|data availability|"
    r"evidence tables|supplementary data|funding|conflict of interest)",
    re.I,
)


def _garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) in _CONTROL) / len(text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _heading_key(text: str) -> str:
    """Normalized text with any leading section numbering stripped.

    The body heading reads '21. References' while the outline entry is plain
    'References', so the numbering has to come off both sides before comparing.
    """
    return _NUM_PREFIX.sub("", _normalize(text))


def _order_blocks(blocks, rect):
    """Put blocks of a two-column page into reading order.

    Sorting by (y, x) interleaves the columns -- a left-column heading and an
    unrelated right-column paragraph share the same y, so the section pointer
    jumps between them. Instead classify each block as left, right, or
    full-width; full-width blocks (wide tables, spanning headings) act as
    horizontal band separators, and within each band the whole left column is
    emitted before the whole right column.
    """
    mid = (rect.x0 + rect.x1) / 2
    full, left, right = [], [], []
    for b in blocks:
        x0, _y0, x1, _y1 = b[0], b[1], b[2], b[3]
        if (x1 - x0) > _FULL_WIDTH_FRAC * rect.width:
            full.append(b)
        elif (x0 + x1) / 2 < mid:
            left.append(b)
        else:
            right.append(b)

    for group in (full, left, right):
        group.sort(key=lambda b: (b[1], b[0]))

    ordered = []
    li = ri = 0
    for fb in full + [None]:
        boundary = fb[1] if fb is not None else float("inf")
        while li < len(left) and left[li][1] < boundary:
            ordered.append(left[li]); li += 1
        while ri < len(right) and right[ri][1] < boundary:
            ordered.append(right[ri]); ri += 1
        if fb is not None:
            ordered.append(fb)
    return ordered


def extract_structured_pdf(
    pdf_path: str,
    drop_back_matter: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Extract text block-by-block, tagging each block with its TOC section.

    Sections come from the PDF's own bookmark outline: for each page we take the
    outline entries starting on it and match them against that page's blocks, so
    a boundary lands on the exact block opening the section rather than on a
    page break.

    Returns (records, stats).
    """
    doc = fitz.open(pdf_path)
    doc_name = os.path.basename(pdf_path)

    toc_by_page: Dict[int, List[str]] = {}
    for _level, title, page_no in doc.get_toc():
        toc_by_page.setdefault(page_no, []).append(title.strip())

    records: List[Dict[str, Any]] = []
    current_section = "Front matter"
    stats = {k: 0 for k in ("blocks_kept", "blocks_garbled", "blocks_margin",
                            "blocks_narrow", "blocks_back_matter", "pages_empty")}

    for page_num, page in enumerate(doc, start=1):
        rect = page.rect
        blocks = [b for b in page.get_text("blocks") if b[4].strip()]
        pending = list(toc_by_page.get(page_num, []))
        kept_on_page = 0

        for block in _order_blocks(blocks, rect):
            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            text = block[4].strip()

            # Running header / footer band.
            if y0 < _MARGIN or y1 > rect.height - _MARGIN:
                stats["blocks_margin"] += 1
                continue

            # Rotated marginalia, e.g. the "Downloaded from academic.oup.com"
            # watermark, which sits mid-page and so survives the margin filter.
            if (x1 - x0) < _MIN_BLOCK_WIDTH:
                stats["blocks_narrow"] += 1
                continue

            # Glyphs that did not decode to real characters.
            if _garbage_ratio(text) > _GARBAGE_THRESHOLD:
                stats["blocks_garbled"] += 1
                continue

            # Does this block open one of the sections starting on this page?
            # A heading is usually the block's first line, with the section's
            # opening paragraph following in the same block, so match on that
            # line rather than on the whole block.
            first_line = text.split("\n", 1)[0]
            if len(first_line) <= _HEADING_MAX_CHARS:
                key = _heading_key(first_line)
                for title in list(pending):
                    tkey = _heading_key(title)
                    if len(tkey) >= 4 and (key.startswith(tkey) or tkey.startswith(key)):
                        current_section = title
                        pending.remove(title)
                        break

            if drop_back_matter and _BACK_MATTER.match(_heading_key(current_section)):
                stats["blocks_back_matter"] += 1
                continue

            records.append({
                "doc_name": doc_name,
                "page": page_num,
                "section": current_section,
                "text": re.sub(r"[ \t]+", " ", text),
            })
            kept_on_page += 1
            stats["blocks_kept"] += 1

        if kept_on_page == 0:
            stats["pages_empty"] += 1

    doc.close()
    return records, stats


# ==========================================
# 2. SECTION-AWARE CHUNKING & METADATA
# ==========================================
def chunk_document(
    records: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split each section's text as one unit, carrying section + page metadata.

    Chunking runs over the whole section (which may span several pages) rather
    than per page, so a paragraph broken across a page boundary stays intact.
    Each chunk's page range is recovered from a char-offset -> page map.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Group consecutive blocks into runs of the same section.
    sections: List[Dict[str, Any]] = []
    for rec in records:
        if not sections or sections[-1]["section"] != rec["section"]:
            sections.append({
                "doc_name": rec["doc_name"],
                "section": rec["section"],
                "blocks": [],
            })
        sections[-1]["blocks"].append((rec["page"], rec["text"]))

    chunks: List[Document] = []
    for sec in sections:
        starts: List[int] = []
        pages: List[int] = []
        parts: List[str] = []
        pos = 0
        for page, text in sec["blocks"]:
            starts.append(pos)
            pages.append(page)
            parts.append(text)
            pos += len(text) + 1  # +1 for the joining newline
        full_text = "\n".join(parts)

        def page_at(offset: int) -> int:
            return pages[max(0, bisect.bisect_right(starts, offset) - 1)]

        cursor = 0
        for split in splitter.split_text(full_text):
            if not split.strip():
                continue
            idx = full_text.find(split, cursor)
            if idx == -1:
                idx = cursor
            cursor = idx + 1

            start_page = page_at(idx)
            end_page = page_at(idx + len(split) - 1)
            chunks.append(Document(
                page_content=split,
                metadata={
                    "source": sec["doc_name"],
                    "section": sec["section"],
                    "page": start_page,
                    "page_end": end_page,
                },
            ))

    return chunks


# ==========================================
# 3. EMBEDDING & VECTOR SYSTEM
# ==========================================
class RAGIndex:
    COLLECTION = "pdf_rag"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        persist_directory: str | None = None,
    ):
        # Local, open-source embedding model. Vectors are normalized and the
        # collection is created with cosine space, so the score Chroma returns
        # is a cosine distance in [0, 2].
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.persist_directory = persist_directory
        self.vector_store = None

    def build_index(self, chunks: List[Document]):
        """Generate embeddings and build the Chroma vector database."""
        if not chunks:
            raise ValueError("No chunks to index -- check PDF extraction.")

        # Chroma appends to an existing collection rather than replacing it, so
        # a rebuild (as the tuning sweep does) has to drop the old one first or
        # every configuration would be searched against the union of all
        # previous ones.
        if self.vector_store is not None:
            self.vector_store.delete_collection()

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.COLLECTION,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self.persist_directory,
        )

    def open_existing(self) -> bool:
        """Attach to an index already on disk instead of re-embedding.

        Returns False if there is nothing persisted yet, so the caller can
        decide to build. Embedding 2000+ blocks takes ~40 s, which is far too
        slow to repeat per Streamlit session.
        """
        if not self.persist_directory or not os.path.isdir(self.persist_directory):
            return False
        store = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        if store._collection.count() == 0:
            return False
        self.vector_store = store
        return True

    def count(self) -> int:
        return 0 if self.vector_store is None else self.vector_store._collection.count()

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Top-k semantic search, returning (document, distance) pairs."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        return self.vector_store.similarity_search_with_score(query, k=k)


# ==========================================
# 4. TUNING AND VISUAL DISPLAY INTERFACE
# ==========================================
def display_retrieved_chunks(query: str, results: List[Tuple[Document, float]]):
    """Display retrieved chunks with clean visual formatting before generation."""
    print("\n" + "=" * 80)
    print(f"USER QUERY: '{query}'")
    print(f"RETRIEVED CHUNKS: {len(results)}")
    print("=" * 80)

    for idx, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        pages = str(m["page"]) if m["page"] == m["page_end"] else f"{m['page']}-{m['page_end']}"
        print(f"\n[Chunk {idx}]  distance={score:.4f}  (lower = closer)")
        print(f"  File    : {m['source']}")
        print(f"  Section : {m['section']}")
        print(f"  Page    : {pages}")
        print("-" * 80)
        print(doc.page_content.strip())
        print("-" * 80)


# ==========================================
# 5 & 6. TUNING: CHUNK SIZE / OVERLAP / K
# ==========================================

# Gold set: a query paired with the section(s) that actually answer it. Labels
# are section titles from the PDF outline, so they can be checked automatically.
# Some questions are legitimately answered by more than one section.
GOLD: List[Tuple[str, Tuple[str, ...]]] = [
    ("How should maternal risk be assessed before pregnancy?",
     ("4.2.1.1.",)),
    ("Is warfarin safe to use during pregnancy?",
     ("5.2.1.1.",)),
    ("Can direct oral anticoagulants be given to pregnant women?",
     ("5.2.1.5.",)),
    ("How is pre-eclampsia diagnosed and managed?",
     ("12.3.3.2.2.", "12.3.2.")),
    ("What is the treatment for venous thromboembolism in pregnancy?",
     ("11.4.2.", "11.4.")),
    ("How should atrial fibrillation be anticoagulated in pregnancy?",
     ("12.4.1.2.",)),
    ("Is breastfeeding safe for women taking cardiac medication?",
     ("13.2.",)),
    ("When is endocarditis antibiotic prophylaxis needed at delivery?",
     ("4.5.7.",)),
    ("How is cardiac arrest managed in a pregnant woman?",
     ("12.4.4.",)),
    ("How is hypertrophic cardiomyopathy managed during pregnancy?",
     ("6.1.3.",)),
    ("What are the risks of ionizing radiation exposure to the foetus?",
     ("4.3.5.",)),
    ("How is long QT syndrome managed in pregnancy?",
     ("6.2.1.",)),
    ("What are the recommendations for cardiac surgery during pregnancy?",
     ("8.7.",)),
    ("How does blood volume change physiologically during pregnancy?",
     ("3.2.",)),
    ("How should beta blockers be used in pregnant patients?",
     ("5.2.9.",)),
    ("What is pregnancy-associated spontaneous coronary artery dissection?",
     ("12.2.1.3.",)),
    ("How is chronic heart failure treated in pregnant women?",
     ("12.6.1.", "12.6.")),
    ("What are the risks of a mechanical heart valve during pregnancy?",
     ("12.5.3.2.2.", "12.5.3.2.1.", "12.5.3.")),
]

K_GRID = (1, 3, 5, 10)
SIZE_GRID = (400, 600, 800, 1000)
OVERLAP_FRACTIONS = (0.10, 0.25)


def _score(index: "RAGIndex", k_max: int) -> Dict[str, float]:
    """Hit@k for each k in K_GRID, plus MRR over the gold set."""
    hits = {k: 0 for k in K_GRID}
    rr_total = 0.0

    for query, accepted in GOLD:
        results = index.search(query, k=k_max)
        rank = None
        for i, (doc, _dist) in enumerate(results, 1):
            if doc.metadata["section"].startswith(accepted):
                rank = i
                break
        if rank is not None:
            rr_total += 1.0 / rank
            for k in K_GRID:
                if rank <= k:
                    hits[k] += 1

    n = len(GOLD)
    out = {f"hit@{k}": hits[k] / n for k in K_GRID}
    out["mrr"] = rr_total / n
    return out


def tune(records: List[Dict[str, Any]]) -> None:
    """Sweep chunk size x overlap, scoring every k in K_GRID per configuration.

    The embedding model is built once and reused: only the chunking and the
    Chroma collection are rebuilt per configuration.
    """
    index = RAGIndex()  # loads the model once
    k_max = max(K_GRID)

    header = (f"{'chunk':>6} {'overlap':>8} {'chunks':>7} {'avg len':>8} "
              + " ".join(f"{'hit@'+str(k):>7}" for k in K_GRID) + f"{'MRR':>8}")
    print("\n" + "=" * len(header))
    print(f"TUNING over {len(GOLD)} labelled queries")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = []
    for size in SIZE_GRID:
        for frac in OVERLAP_FRACTIONS:
            overlap = int(size * frac)
            chunks = chunk_document(records, chunk_size=size, chunk_overlap=overlap)
            avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
            index.build_index(chunks)
            m = _score(index, k_max)
            rows.append((size, overlap, m))
            print(f"{size:>6} {overlap:>8} {len(chunks):>7} {avg_len:>8.0f} "
                  + " ".join(f"{m['hit@'+str(k)]:>7.2f}" for k in K_GRID)
                  + f"{m['mrr']:>8.3f}")

    best = max(rows, key=lambda r: (r[2]["mrr"], r[2][f"hit@{K_GRID[1]}"]))
    print("-" * len(header))
    print(f"BEST: chunk_size={best[0]} chunk_overlap={best[1]} "
          f"(MRR={best[2]['mrr']:.3f})")

    # Smallest k that is within 1 gold query of the best hit rate at this config.
    hits = [best[2][f"hit@{k}"] for k in K_GRID]
    ceiling = max(hits)
    knee = next(k for k, h in zip(K_GRID, hits) if h >= ceiling - 1.0 / len(GOLD))
    print(f"      k={knee} reaches hit@k={best[2][f'hit@{knee}']:.2f} "
          f"(ceiling {ceiling:.2f} at k={K_GRID[hits.index(ceiling)]})")
    print("=" * len(header))


# ==========================================
# EXECUTION & TUNING WORKFLOW
# ==========================================
if __name__ == "__main__":
    pdf_filename = PDF_PATH

    if not os.path.exists(pdf_filename):
        sys.exit(f"PDF not found: {pdf_filename}")

    # ---- Tunables (steps 5 & 6) -------------------------------------------
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    K_VALUE = 4
    USER_QUERY = "How should maternal risk be assessed before pregnancy?"
    # -----------------------------------------------------------------------

    # Step 1: structured parse
    records, stats = extract_structured_pdf(pdf_filename)
    print(f"Parsed {stats['blocks_kept']} blocks  (dropped: "
          f"{stats['blocks_garbled']} garbled, {stats['blocks_margin']} header/footer, "
          f"{stats['blocks_narrow']} marginalia, {stats['blocks_back_matter']} back matter)")

    # Steps 5 & 6: sweep chunk size / overlap / k instead of a single run
    if "--tune" in sys.argv:
        tune(records)
        raise SystemExit(0)

    # Steps 2 & 3: section-aware chunking + metadata binding
    chunks = chunk_document(records, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Built {len(chunks)} chunks across "
          f"{len({c.metadata['section'] for c in chunks})} sections")

    # Step 2 (cont): embed + index. `--build` writes the index to disk so the
    # Streamlit app can attach to it instead of re-embedding every session.
    if "--build" in sys.argv:
        rag_system = RAGIndex(persist_directory=PERSIST_DIR)
        rag_system.build_index(chunks)
        print(f"Persisted {rag_system.count()} chunks to {PERSIST_DIR}")
        raise SystemExit(0)

    rag_system = RAGIndex()
    rag_system.build_index(chunks)
    print("Chroma index ready.")

    # Step 4: top-k semantic search
    results = rag_system.search(USER_QUERY, k=K_VALUE)

    # Step 7: display retrieved chunks before generation
    display_retrieved_chunks(USER_QUERY, results)
