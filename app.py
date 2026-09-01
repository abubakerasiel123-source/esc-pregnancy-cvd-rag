"""Streamlit front end for the pregnancy/CVD guideline RAG system.

Run it however is convenient -- `streamlit run app.py`, `python app.py`, or
the editor Run button -- and the bootstrap below sorts out the rest.

Requires a persisted index; build it once with:
    python biko.py --build
"""
import os
import sys
import subprocess

# --- interpreter / launcher bootstrap (must precede third-party imports) ---
_INTERPRETERS = (
    "/opt/anaconda3/envs/rag/bin/python",
    "/opt/anaconda3/bin/python",
)
_PROBE = "import streamlit"


def _serve_with_streamlit() -> None:
    """Re-launch this file under `streamlit run` unless it already is.

    Two different failure modes end here. `python app.py` on an interpreter
    without streamlit dies at the import; on one *with* streamlit it survives
    the import but renders nothing, because a Streamlit script only produces a
    page when the streamlit runtime is serving it. Both are fixed by the same
    re-exec, so neither is worth telling apart.
    """
    if os.environ.get("_RAG_APP_RELAUNCHED"):
        return  # we already re-exec'd once -- never loop

    import importlib.util
    if importlib.util.find_spec("streamlit") is not None:
        from streamlit.runtime import exists as _runtime_exists
        if _runtime_exists():
            return  # being served already: the normal path, costs one import

    # sys.executable first, so a correctly-chosen interpreter is kept and only
    # the launcher changes.
    for candidate in (sys.executable, *_INTERPRETERS):
        if not os.path.exists(candidate):
            continue
        if subprocess.run([candidate, "-c", _PROBE], capture_output=True).returncode:
            continue
        print(f"[bootstrap] serving app.py with {candidate} -m streamlit run\n",
              file=sys.stderr)
        os.execve(
            candidate,
            [candidate, "-m", "streamlit", "run", os.path.abspath(__file__),
             *sys.argv[1:]],
            {**os.environ, "_RAG_APP_RELAUNCHED": "1"},
        )

    sys.exit(
        "streamlit is not installed in any interpreter this script knows of.\n"
        "Install it into the env you want to use, e.g.:\n"
        "  conda activate rag\n"
        "  pip install streamlit"
    )


_serve_with_streamlit()
# ---------------------------------------------------------------------------

import streamlit as st

import config  # noqa: F401 -- populates os.environ from .env on import
import generator
from biko import PERSIST_DIR, RAGIndex

st.set_page_config(page_title="ESC Pregnancy & CVD Guidelines", page_icon="🫀",
                   layout="wide")


@st.cache_resource(show_spinner="Loading index...")
def load_index() -> RAGIndex:
    """Attach to the on-disk Chroma index once per server process.

    cache_resource (not cache_data) because the returned object holds an open
    DB handle and a loaded embedding model, neither of which is serializable.
    """
    index = RAGIndex(persist_directory=PERSIST_DIR)
    if not index.open_existing():
        return None
    return index


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    # st.secrets raises rather than returning empty when no secrets.toml exists.
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Retrieval settings")
    k = st.slider("Chunks retrieved (k)", 1, 10, 5,
                  help="Tuning showed hit@k reaches 1.00 at k=5 on the "
                       "labelled gold set; k=3 already reaches 0.94.")
    show_sources = st.toggle("Show retrieved chunks", value=True,
                             help="Requirement 7: inspect what was retrieved "
                                  "before generation.")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "Answers are generated only from text retrieved from the 2025 ESC "
        "Guidelines PDF. This is a document-lookup tool, **not medical "
        "advice**, and it can misread or miss content. Verify against the "
        "source document before acting on anything here."
    )

st.title("🫀 ESC Pregnancy & Cardiovascular Disease Guidelines")
st.caption("Ask a question about the 2025 ESC guideline. Answers cite the "
           "section and page they came from.")

index = load_index()
if index is None:
    st.error(
        f"No index found at `{PERSIST_DIR}`.\n\n"
        "Build it first:\n\n"
        "```bash\n/opt/anaconda3/envs/rag/bin/python biko.py --build\n```"
    )
    st.stop()

if not has_credentials():
    st.warning(
        "`ANTHROPIC_API_KEY` is not set, so retrieval works but answers "
        "cannot be generated. Set it in your shell, or add it to "
        "`.streamlit/secrets.toml`."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------------------------------
# Replay history
# --------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['n']}] {src['section']}** — page {src['pages']} "
                        f"· distance {src['distance']:.3f}"
                    )
                    st.text(src["text"])

# --------------------------------------------------------------------------
# New turn
# --------------------------------------------------------------------------
if prompt := st.chat_input("e.g. Is warfarin safe during pregnancy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Step 4: retrieve
        with st.spinner("Searching the guideline..."):
            results = index.search(prompt, k=k)

        sources = []
        for i, (doc, dist) in enumerate(results, 1):
            m = doc.metadata
            sources.append({
                "n": i,
                "section": m["section"],
                "pages": (str(m["page"]) if m["page"] == m["page_end"]
                          else f"{m['page']}-{m['page_end']}"),
                "distance": dist,
                "text": doc.page_content.strip(),
            })

        # Step 7: show what was retrieved, before generation
        if show_sources and sources:
            with st.expander(f"{len(sources)} chunks retrieved", expanded=False):
                for src in sources:
                    st.markdown(
                        f"**[{src['n']}] {src['section']}** — page {src['pages']} "
                        f"· distance {src['distance']:.3f}"
                    )
                    st.text(src["text"])

        if generator.is_out_of_scope(results):
            answer = (
                "That question doesn't appear to be covered by this document. "
                "It contains the 2025 ESC Guidelines on cardiovascular disease "
                "and pregnancy — try asking about risk assessment, "
                "anticoagulation, delivery planning, or specific cardiac "
                "conditions in pregnancy."
            )
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
        elif not has_credentials():
            answer = ("Retrieved the sources above, but `ANTHROPIC_API_KEY` is "
                      "not set so no answer could be generated.")
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        else:
            # Prior turns only -- each turn carries its own freshly retrieved
            # sources, so replaying old source blocks would just crowd context.
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                answer = st.write_stream(
                    generator.answer_stream(prompt, results, history)
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:  # surface API errors in the UI, not the console
                st.error(f"Generation failed: {type(exc).__name__}: {exc}")
