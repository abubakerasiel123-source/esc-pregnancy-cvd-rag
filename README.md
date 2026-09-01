# ESC Pregnancy & Cardiovascular Disease — RAG

A retrieval-augmented question answering system over the **2025 ESC Guidelines
for the management of cardiovascular disease during pregnancy**. Ask a clinical
question; get an answer assembled only from passages retrieved out of the
guideline, with the section and page each claim came from.

> This is a **document-lookup tool, not medical advice**. It reports what the
> guideline says, it can misread or miss content, and every answer should be
> verified against the source document.

## How it works

| Stage | Implementation |
|---|---|
| Parse | PyMuPDF, section-aware. Sections come from the PDF's own bookmark outline; a two-column reading-order pass keeps left and right columns from interleaving; pages whose embedded fonts decode to control characters are dropped rather than indexed. |
| Chunk | `RecursiveCharacterTextSplitter`, carrying section title and page span through as metadata. |
| Embed | `BAAI/bge-small-en-v1.5` locally — no API key, no data leaves the machine. |
| Store | Chroma, persisted to disk (726 chunks). |
| Retrieve | Cosine top-*k*, default *k*=5. |
| Generate | Claude (`claude-opus-5`) under a strict grounding contract: answer only from the numbered sources, cite inline for every clinical claim, preserve recommendation classes, evidence levels, doses and gestational timing verbatim, and say what is missing rather than guess. |

Retrieval is deliberately gated. Below a cosine-distance cutoff of `0.32` the
question is treated as out of scope and **no API request is sent at all** — on
the labelled set, in-scope questions top out at 0.225 and out-of-scope ones
start at 0.411, so the two clusters separate cleanly.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

The guideline PDF is **not** included — it is copyrighted by the European
Society of Cardiology. Download it from the ESC and save it in the repo root as
`pregnancy_cvd_2025.pdf`.

Build the index once (a few minutes; embeds locally):

```bash
python biko.py --build
```

Then run the app:

```bash
streamlit run app.py
```

`app.py` re-launches itself under `streamlit run` with an interpreter that has
the dependencies, so `python app.py` and the editor Run button work too.

Retrieval works without an API key; only generation needs one.

## Evaluation

Two harnesses, measuring different things.

**Retrieval** — `python biko.py --tune` scores hit@k against 18 labelled
questions whose accepted answers are identified by section title. hit@k reaches
**1.00 at k=5**; k=3 already reaches 0.94.

**Generation** — `python evaluate.py` measures whether the answer built from
those chunks is *faithful*: that every substantive sentence carries a citation,
that no citation points at a source that was not supplied, and that
out-of-scope questions are refused rather than answered. Every question is a
billed API call.

## Files

| File | Role |
|---|---|
| `biko.py` | Parsing, chunking, embedding, index, retrieval, `--build` / `--tune`. |
| `generator.py` | Grounded generation: the system contract, source formatting, scope gate. |
| `app.py` | Streamlit chat UI, with retrieved chunks inspectable before the answer. |
| `evaluate.py` | Citation-faithfulness harness. |
| `config.py` | Loads `.env` so every entry point sees the same credentials. |
| `Pregnancy_Rag_System.py` | Earlier single-file prototype, kept for reference. |
