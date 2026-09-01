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

import html

import streamlit as st

import config  # noqa: F401 -- populates os.environ from .env on import
import generator
from biko import PERSIST_DIR, RAGIndex

st.set_page_config(page_title="ESC Pregnancy & CVD Guidelines", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Design system
#
# Styling leans on custom HTML we own (hero, cards, pills) rather than on
# Streamlit's internal class names, which churn between releases. The handful
# of selectors below use data-testid attributes, the most stable hooks
# Streamlit exposes.
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --rose:      #C0334D;
  --rose-dark: #97263B;
  --rose-soft: #FBEAEE;
  --teal:      #2F7D8C;
  --ink:       #241F26;
  --muted:     #6E6675;
  --line:      #EAE3E6;
  --paper:     #FFFFFF;
}

html, body, [class*="st-"], button, input, textarea {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
               Helvetica, Arial, sans-serif;
}

/* Wide layout, but text should not run the full width of a 27" display. */
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* ---------- Hero ---------- */
.hero {
  display: flex; align-items: center; gap: 28px;
  background: linear-gradient(135deg, #FFF7F8 0%, #F6F1FA 55%, #EFF6F7 100%);
  border: 1px solid var(--line); border-radius: 20px;
  padding: 26px 30px; margin-bottom: 22px;
}
.hero-copy { flex: 1 1 340px; min-width: 260px; }
.hero-art  { flex: 0 1 400px; min-width: 220px; line-height: 0; }
.hero h1 {
  font-size: 1.72rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ink); margin: 0 0 8px 0; line-height: 1.22;
}
.hero p { color: var(--muted); font-size: 0.95rem; margin: 0 0 14px 0; line-height: 1.5; }
.pill-row { display: flex; flex-wrap: wrap; gap: 7px; }
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(255,255,255,0.85); border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 11px;
  font-size: 0.735rem; font-weight: 500; color: var(--muted);
}
.pill b { color: var(--ink); font-weight: 600; }
@media (max-width: 780px) {
  .hero { flex-direction: column; align-items: flex-start; }
  .hero-art { display: none; }
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.side-title {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--muted); margin: 4px 0 10px 0;
}
.status {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.83rem; color: var(--ink); padding: 7px 0;
}
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot.ok   { background: #2E9E6B; box-shadow: 0 0 0 3px rgba(46,158,107,0.15); }
.dot.warn { background: #D9922B; box-shadow: 0 0 0 3px rgba(217,146,43,0.15); }
.status .meta { color: var(--muted); margin-left: auto; font-size: 0.78rem; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
  background: var(--paper); border: 1px solid var(--line);
  border-radius: 16px; padding: 14px 18px; margin-bottom: 12px;
  box-shadow: 0 1px 2px rgba(36,31,38,0.04);
}
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }

/* ---------- Source cards ---------- */
.src {
  border: 1px solid var(--line); border-left: 3px solid var(--rose);
  border-radius: 12px; background: var(--paper);
  padding: 12px 14px; margin-bottom: 10px;
}
.src-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.src-n {
  background: var(--rose-soft); color: var(--rose-dark);
  font-weight: 700; font-size: 0.72rem;
  border-radius: 6px; padding: 2px 7px; flex: none;
}
.src-sec { font-weight: 600; font-size: 0.87rem; color: var(--ink); flex: 1 1 auto; }
.src-page {
  font-size: 0.72rem; color: var(--muted);
  background: #F5F2F4; border-radius: 999px; padding: 2px 9px; flex: none;
}
.src-body {
  font-size: 0.815rem; line-height: 1.55; color: #443C48;
  white-space: pre-wrap; max-height: 210px; overflow-y: auto;
  background: #FCFAFB; border-radius: 8px; padding: 10px 12px;
}
.gauge { display: flex; align-items: center; gap: 8px; margin-top: 9px; }
.gauge-track { flex: 1; height: 4px; background: #F0EAED; border-radius: 999px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--teal), var(--rose)); }
.gauge-label { font-size: 0.7rem; color: var(--muted); font-variant-numeric: tabular-nums; flex: none; }

/* ---------- Empty state ---------- */
.empty-hint {
  font-size: 0.76rem; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--muted); margin: 6px 0 10px 2px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# The illustration is this system, not stock art: the guideline PDF on the
# left, the chunks it is split into, and the cited answer that comes back --
# with the cardiac motif that makes the subject matter obvious at a glance.
HERO_ART = """
<svg viewBox="0 0 420 210" role="img" width="100%"
     aria-label="Diagram: the ESC guideline PDF is split into chunks, retrieved, and returned as a cited answer">
  <defs>
    <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#FAF6F8"/>
    </linearGradient>
    <filter id="sh" x="-25%" y="-25%" width="150%" height="150%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#241F26" flood-opacity="0.11"/>
    </filter>
  </defs>

  <!-- 1. the source document -->
  <g filter="url(#sh)">
    <rect x="14" y="34" width="104" height="132" rx="10" fill="url(#pg)" stroke="#EAE3E6"/>
  </g>
  <rect x="14" y="34" width="104" height="26" rx="10" fill="#C0334D" opacity="0.10"/>
  <path d="M26 47h58" stroke="#C0334D" stroke-width="3.5" stroke-linecap="round"/>
  <g stroke="#CFC6CC" stroke-width="2.6" stroke-linecap="round">
    <path d="M26 76h80"/><path d="M26 89h80"/><path d="M26 102h62"/>
    <path d="M26 115h80"/><path d="M26 128h71"/><path d="M26 141h44"/>
  </g>
  <text x="66" y="185" text-anchor="middle" font-size="11" font-weight="600"
        fill="#6E6675" font-family="Inter, sans-serif">2025 ESC PDF</text>

  <!-- flow -->
  <path d="M124 100h28" stroke="#C9BFC6" stroke-width="2" stroke-dasharray="4 4" stroke-linecap="round"/>

  <!-- 2. chunks + embeddings -->
  <g filter="url(#sh)">
    <rect x="158" y="56" width="86" height="26" rx="7" fill="#FFFFFF" stroke="#EAE3E6"/>
    <rect x="158" y="88" width="86" height="26" rx="7" fill="#FFFFFF" stroke="#2F7D8C" stroke-width="1.6"/>
    <rect x="158" y="120" width="86" height="26" rx="7" fill="#FFFFFF" stroke="#EAE3E6"/>
  </g>
  <g fill="#2F7D8C">
    <circle cx="170" cy="101" r="3"/><circle cx="182" cy="101" r="3"/>
    <circle cx="194" cy="101" r="3"/><circle cx="206" cy="101" r="3"/>
  </g>
  <g stroke="#DCD4D9" stroke-width="2.4" stroke-linecap="round">
    <path d="M170 69h50"/><path d="M170 133h50"/>
  </g>
  <text x="201" y="185" text-anchor="middle" font-size="11" font-weight="600"
        fill="#6E6675" font-family="Inter, sans-serif">chunks + vectors</text>

  <!-- flow -->
  <path d="M250 100h26" stroke="#C9BFC6" stroke-width="2" stroke-dasharray="4 4" stroke-linecap="round"/>

  <!-- 3. the cited answer, with the cardiac motif -->
  <g filter="url(#sh)">
    <rect x="282" y="40" width="124" height="120" rx="12" fill="#FFFFFF" stroke="#EAE3E6"/>
  </g>
  <path d="M344 74c-9-13-27-8-27 6 0 11 16 20 27 30 11-10 27-19 27-30 0-14-18-19-27-6z"
        fill="#C0334D" opacity="0.13"/>
  <path d="M298 92h14l6-13 9 25 7-16 5 8h14" fill="none" stroke="#C0334D"
        stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
  <g stroke="#CFC6CC" stroke-width="2.6" stroke-linecap="round">
    <path d="M298 126h92"/><path d="M298 138h64"/>
  </g>
  <g font-family="Inter, sans-serif" font-size="9" font-weight="700" fill="#97263B">
    <rect x="296" y="146" width="19" height="13" rx="4" fill="#FBEAEE"/>
    <text x="305.5" y="155.5" text-anchor="middle">[1]</text>
    <rect x="319" y="146" width="19" height="13" rx="4" fill="#FBEAEE"/>
    <text x="328.5" y="155.5" text-anchor="middle">[2]</text>
  </g>
  <text x="344" y="185" text-anchor="middle" font-size="11" font-weight="600"
        fill="#6E6675" font-family="Inter, sans-serif">cited answer</text>
</svg>
"""

EXAMPLES = [
    "How should maternal risk be assessed before pregnancy?",
    "Is warfarin safe to use during pregnancy?",
    "What are the risks of a mechanical heart valve during pregnancy?",
    "How is pre-eclampsia diagnosed and managed?",
]


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


def resolve_credentials() -> bool:
    """Report whether a key is available, exporting it to the environment.

    The Anthropic client reads ANTHROPIC_API_KEY from os.environ, but on
    Streamlit Community Cloud the key arrives through st.secrets instead.
    Without copying it across, the sidebar would report generation "ready" and
    every answer would then fail with an authentication error.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    # st.secrets raises rather than returning empty when no secrets.toml exists.
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return False
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        return True
    return False


def render_sources(sources, label: str, expanded: bool = False) -> None:
    """Draw retrieved chunks as cards, with distance shown against the cutoff."""
    with st.expander(label, expanded=expanded):
        for src in sources:
            # Full bar at distance 0, empty at the out-of-scope cutoff. The raw
            # number stays on screen -- the bar is an aid, not a replacement.
            fill = max(0.0, min(1.0, 1 - src["distance"] / generator.RELEVANCE_CUTOFF))
            st.markdown(
                f'<div class="src">'
                f'  <div class="src-head">'
                f'    <span class="src-n">{src["n"]}</span>'
                f'    <span class="src-sec">{html.escape(src["section"])}</span>'
                f'    <span class="src-page">page {html.escape(str(src["pages"]))}</span>'
                f'  </div>'
                f'  <div class="src-body">{html.escape(src["text"])}</div>'
                f'  <div class="gauge">'
                f'    <div class="gauge-track"><div class="gauge-fill" style="width:{fill*100:.0f}%"></div></div>'
                f'    <span class="gauge-label">distance {src["distance"]:.3f}</span>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )


index = load_index()
credentials = resolve_credentials()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">Status</div>', unsafe_allow_html=True)
    if index is None:
        st.markdown('<div class="status"><span class="dot warn"></span>'
                    'Index missing</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status"><span class="dot ok"></span>Index loaded'
                    f'<span class="meta">{index.count()} chunks</span></div>',
                    unsafe_allow_html=True)
    if credentials:
        st.markdown('<div class="status"><span class="dot ok"></span>Generation'
                    '<span class="meta">ready</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status"><span class="dot warn"></span>Generation'
                    '<span class="meta">no API key</span></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="side-title">Retrieval</div>', unsafe_allow_html=True)
    k = st.slider("Chunks retrieved (k)", 1, 10, 5,
                  help="Tuning showed hit@k reaches 1.00 at k=5 on the "
                       "labelled gold set; k=3 already reaches 0.94.")
    show_sources = st.toggle("Show retrieved chunks", value=True,
                             help="Inspect what was retrieved before generation.")

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

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f'<div class="hero">'
    f'  <div class="hero-copy">'
    f'    <h1>ESC Pregnancy &amp; Cardiovascular Disease</h1>'
    f'    <p>Ask a question about the 2025 ESC guideline. Every answer is built '
    f'       only from retrieved passages and cites the section and page it '
    f'       came from.</p>'
    f'    <div class="pill-row">'
    f'      <span class="pill">🫀 <b>2025 ESC</b> guideline</span>'
    f'      <span class="pill">🔎 <b>bge-small</b> embeddings</span>'
    f'      <span class="pill">📎 <b>cited</b> answers</span>'
    f'    </div>'
    f'  </div>'
    f'  <div class="hero-art">{HERO_ART}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

if index is None:
    st.error(
        f"No index found at `{PERSIST_DIR}`.\n\n"
        "Build it first:\n\n"
        "```bash\npython biko.py --build\n```"
    )
    st.stop()

if not credentials:
    st.warning(
        "`ANTHROPIC_API_KEY` is not set, so retrieval works but answers "
        "cannot be generated. Add it to `.env`, set it in your shell, or put "
        "it in `.streamlit/secrets.toml`."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------------------------------
# Empty state -- seed the first question instead of facing a blank page
# --------------------------------------------------------------------------
# Held in a placeholder so the chips can be cleared the moment a question is
# asked, without a rerun -- a rerun here would also discard the "chunks
# retrieved" panel drawn during the turn.
chips_slot = st.empty()
if not st.session_state.messages:
    with chips_slot.container():
        st.markdown('<div class="empty-hint">Try one of these</div>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()

# --------------------------------------------------------------------------
# Replay history
# --------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"], f"{len(msg['sources'])} sources")

# --------------------------------------------------------------------------
# New turn -- from the chat box, or from an example chip
# --------------------------------------------------------------------------
prompt = st.chat_input("Ask about risk, anticoagulation, delivery planning…")
if not prompt:
    prompt = st.session_state.pop("pending", None)

if prompt:
    chips_slot.empty()
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
            render_sources(sources, f"{len(sources)} chunks retrieved")

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
        elif not credentials:
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
