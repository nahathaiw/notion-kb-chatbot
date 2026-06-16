"""
Notion KB Chatbot — Streamlit web app.

This wraps the same from-scratch RAG pipeline in `notion_kb_chatbot.py` in a
chat UI you can actually share and demo, instead of a terminal loop. It adds
three things a "real app" needs that the CLI didn't have:

  1. The index is built ONCE and cached (st.cache_resource), so we don't re-embed
     every document on every interaction — only when the docs actually change.
  2. Conversation memory — follow-up questions see the prior turns.
  3. A visible "retrieved chunks" panel under each answer, so you can watch
     retrieval working (the same transparency the CLI printed to stdout).

Run it with:
    export OPENAI_API_KEY="sk-..."
    streamlit run app.py
"""

import os

import streamlit as st

# Load OPENAI_API_KEY (and any other vars) from a local .env file if present, so
# the secret lives in one gitignored place instead of being hardcoded in source.
# A real exported environment variable still takes precedence over .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; export the env var manually if it's not installed.

st.set_page_config(page_title="Notion KB Chatbot", page_icon="📚", layout="centered")

# ---------------------------------------------------------------------------
# Secrets bridge: locally we read keys from .env (above). When deployed on
# Streamlit Community Cloud there is no .env — secrets come from st.secrets
# (set in the app's dashboard). Copy any we find into os.environ so the core
# module, which reads os.environ, works identically in both places. Env vars
# that are already set win, so local .env still takes precedence.
#
# NOTE: touching st.secrets raises FileNotFoundError when no secrets.toml
# exists (the normal local case), so we read it through a guarded helper.
# ---------------------------------------------------------------------------
def _secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default


for _key in ("OPENAI_API_KEY", "NOTION_API_KEY", "NOTION_TOKEN"):
    if not os.environ.get(_key):
        _val = _secret(_key)
        if _val:
            os.environ[_key] = _val


# ---------------------------------------------------------------------------
# Password gate: this app can reach your private Notion workspace and spends
# your OpenAI credits, so it must not be open to the public. We require a
# password (set as APP_PASSWORD in .env locally / in st.secrets on Cloud). If
# no password is configured, the gate is skipped (handy for purely local use).
# ---------------------------------------------------------------------------
def _check_password():
    expected = os.environ.get("APP_PASSWORD") or _secret("APP_PASSWORD", "")
    if not expected:
        return True  # no password configured -> open (local dev convenience)
    if st.session_state.get("authenticated"):
        return True

    st.title("📚 Notion KB Chatbot")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        if st.form_submit_button("Enter") and attempt:
            if attempt == expected:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Guard: the OpenAI client is constructed at import time inside the core module
# and raises if no API key is present. Check first and show a friendly message
# instead of a stack trace.
# ---------------------------------------------------------------------------
if not os.environ.get("OPENAI_API_KEY"):
    st.title("📚 Notion KB Chatbot")
    st.error(
        "No `OPENAI_API_KEY` found. Set it before launching:\n\n"
        "```bash\nexport OPENAI_API_KEY=\"sk-...\"\nstreamlit run app.py\n```"
    )
    st.stop()

# Safe to import now — this also constructs the shared OpenAI client.
import notion_kb_chatbot as rag


# ---------------------------------------------------------------------------
# Index: build once, reuse across reruns.
# ---------------------------------------------------------------------------
# Streamlit reruns the whole script on every interaction. Without caching we'd
# re-load, re-chunk, and re-embed the docs on every message — slow and costly.
# st.cache_resource keeps the built VectorStore alive across reruns; the index
# only rebuilds when the docs' signature (filenames + mtimes) changes, which we
# pass in as a cache key so editing/adding a doc transparently re-indexes.
import glob


def _docs_signature(docs_dir=rag.DOCS_DIR):
    """A cheap fingerprint of the docs folder: (name, mtime) per .txt file."""
    return tuple(
        (os.path.basename(p), os.path.getmtime(p))
        for p in sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    )


@st.cache_resource(show_spinner="Indexing documents…")
def get_store(source, _signature):
    # `_signature` is unused inside, but its value is part of the cache key:
    # change the source's contents and the signature changes, so the index
    # rebuilds. `source` is also part of the key, so each source caches separately.
    if source == "Notion workspace":
        import notion_loader
        documents = notion_loader.fetch_notion_documents()
        return rag.build_index(documents)
    return rag.build_index()  # local docs/ folder


# ---------------------------------------------------------------------------
# Sidebar — settings and visibility into the knowledge base.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    notion_ready = bool(os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN"))
    source = st.radio(
        "Knowledge source",
        options=["Local docs", "Notion workspace"],
        help="Local reads docs/*.txt. Notion pulls live pages your integration "
             "can see (needs NOTION_API_KEY in .env).",
    )
    if source == "Notion workspace" and not notion_ready:
        st.warning(
            "Set `NOTION_API_KEY` in `.env` and share pages with your integration. "
            "See `notion_loader.py` for setup."
        )
        st.stop()

    top_k = st.slider(
        "Chunks to retrieve (top-k)", min_value=1, max_value=8, value=rag.TOP_K,
        help="How many document chunks to feed the model as context per question.",
    )
    st.caption(f"Embedding model: `{rag.EMBED_MODEL}`")
    st.caption(f"Chat model: `{rag.CHAT_MODEL}`")

# Build (or fetch cached) the index for the selected source. For Notion we can't
# cheaply fingerprint remote content, so the cache key is just the source name;
# use the Re-index button to force a fresh pull.
signature = _docs_signature() if source == "Local docs" else "notion"
store = get_store(source, signature)

with st.sidebar:
    st.divider()
    st.subheader("📂 Knowledge base")
    sources = sorted({c["source"] for c in store.chunks})
    st.caption(f"{len(store.chunks)} chunks from {len(sources)} sources:")
    for s in sources:
        st.markdown(f"- `{s}`")

    st.divider()
    if st.button("🔄 Re-index", use_container_width=True):
        get_store.clear()
        st.rerun()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat — render history, then handle the new question.
# ---------------------------------------------------------------------------
st.title("📚 Notion KB Chatbot")
st.caption("Ask about Notion. Answers are grounded in the docs and cite their source.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "retrieved"?}

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved"):
            with st.expander("🔍 Retrieved chunks"):
                for c in msg["retrieved"]:
                    st.markdown(f"**`{c['source']}`** · similarity `{c['score']:.3f}`")
                    st.text(c["text"])

query = st.chat_input("Ask a question…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            retrieved = store.retrieve(query, top_k=top_k)
            # Pass prior turns (content only) so follow-ups have conversational
            # context. We exclude the just-added user turn — generate_answer
            # appends the current question itself.
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            answer = rag.generate_answer(query, retrieved, history=history)
        st.markdown(answer)
        with st.expander("🔍 Retrieved chunks"):
            for c in retrieved:
                st.markdown(f"**`{c['source']}`** · similarity `{c['score']:.3f}`")
                st.text(c["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "retrieved": retrieved}
    )
