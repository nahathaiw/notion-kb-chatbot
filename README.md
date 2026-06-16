# 📚 Notion Knowledge Base Chatbot

A small, readable **Retrieval-Augmented Generation (RAG)** chatbot that answers
questions about your documents — either local `.txt` files **or your real Notion
workspace** — and cites its sources for every claim.

It started as a from-scratch teaching demo (just `openai` + `numpy`, every step
visible) and grew into a real app: a Streamlit chat UI, live Notion ingestion,
a password gate, and one-click deployment.

---

## ✨ Features

- **Chat UI** (Streamlit) with conversation memory and a per-answer panel that
  shows exactly which chunks were retrieved and how similar each was.
- **Two knowledge sources**, switchable in the sidebar:
  - **Local docs** — every `.txt` file in `docs/`.
  - **Notion workspace** — live pages pulled through the Notion API.
- **Grounded answers** — the model is told to answer *only* from retrieved
  context and to cite the source of every claim, so answers are checkable.
- **Cached index** — documents are embedded once and reused, not re-embedded on
  every interaction.
- **Password gate** for safe hosting, plus clean local/Cloud secret handling.
- **From-scratch RAG** — the whole pipeline is ~300 readable lines, no vector DB.

---

## 🧠 How it works

The RAG pipeline, each step a clearly separated function in `notion_kb_chatbot.py`:

| Step | Function | What it does |
|------|----------|--------------|
| Load     | `load_documents` / `notion_loader.fetch_notion_documents` | Read `.txt` files or pull Notion pages, keeping each source for citation |
| Chunk    | `chunk_text` / `build_chunks` | Split docs into overlapping ~800-char windows |
| Embed    | `embed_texts` | Convert chunks to normalized vectors in one batched API call |
| Store    | `VectorStore` | Hold the vectors in memory as a NumPy array |
| Retrieve | `VectorStore.retrieve` | Embed the question, score all chunks by cosine similarity, return top-k |
| Generate | `generate_answer` | Pass the chunks to GPT with a grounded, citation-required prompt |

**Models used:** embeddings `text-embedding-3-small`, generation `gpt-4o-mini`.

---

## 🚀 Quick start

Requires Python 3.8+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your keys (copy the template, then edit .env)
cp .env.example .env
#    -> open .env and paste your real OpenAI key (Notion key is optional)

# 3. Run the web app
./run.sh
```

The app serves at **http://localhost:8502**. Switch the **Knowledge source** in
the sidebar, type a question, and read the answer with its cited sources.

> `run.sh` frees only this project's port (8502) before starting, so it never
> disturbs other local Streamlit apps. The port is pinned in
> `.streamlit/config.toml`, so a plain `streamlit run app.py` uses 8502 too.

### Configuration (`.env`)

```bash
OPENAI_API_KEY=sk-...        # required — embeddings + generation
NOTION_API_KEY=ntn_...       # optional — only for the "Notion workspace" source
APP_PASSWORD=change-me       # optional — gates the app (required before hosting)
```

`.env` is gitignored and never committed.

---

## 🖥️ Two ways to run

**Web app (recommended):**

```bash
./run.sh                 # serves on http://localhost:8502
```

**Command-line loop** (the original minimal demo):

```bash
python notion_kb_chatbot.py
```

Type `quit` to exit the CLI. After each answer it prints the retrieved chunks and
their similarity scores so you can watch retrieval working.

---

## 🗂️ Adding your own data

### Local docs
Drop any `.txt` files into `docs/` and re-run (or click **🔄 Re-index** in the
sidebar). The filename becomes the citation label, so name files descriptively.

### Connect your real Notion workspace
1. Create an internal integration at <https://www.notion.so/my-integrations> and
   copy its **Internal Integration Secret**.
2. Put it in `.env`: `NOTION_API_KEY=ntn_...`
3. **Share pages with the integration**: open a Notion page → **•••** →
   **Connections** → add your integration. *The API only sees pages you share —
   this is the most common first-time gotcha.*
4. In the sidebar, set **Knowledge source → Notion workspace**. Use **🔄 Re-index**
   to pull the latest pages.

`notion_loader.py` searches every accessible page, walks each page's block tree
to extract the text, and feeds it to the same pipeline as local docs.

---

## ☁️ Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (already done if you cloned it from there).
2. Go to <https://share.streamlit.io>, sign in with GitHub, and **Create app**
   from this repo (`main` branch, `app.py`).
3. In **Advanced settings → Secrets**, paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   NOTION_API_KEY = "ntn_..."
   APP_PASSWORD   = "pick-a-strong-password"
   ```
4. **Deploy.** The app is reachable by URL but protected by your password.

> ⚠️ **Security:** This app can read your private Notion pages and spends your
> OpenAI credits, so always set `APP_PASSWORD` before hosting. Keys live only in
> `.env` (local) or Streamlit Secrets (Cloud) — never in the repo.

---

## 📁 Project structure

```
.
├── app.py                  # Streamlit web app (UI, caching, password gate, source switch)
├── notion_kb_chatbot.py    # Core RAG pipeline + CLI
├── notion_loader.py        # Fetch documents live from the Notion API
├── docs/                   # Local .txt knowledge base
│   ├── getting_started.txt
│   ├── databases.txt
│   ├── sharing_permissions.txt
│   ├── documentation_and_wikis.txt
│   └── notion_api_overview.txt
├── requirements.txt
├── .env.example            # template — copy to .env
└── .streamlit/
    └── secrets.toml.example
```

---

## 🛠️ Troubleshooting

- **"No `OPENAI_API_KEY` found"** — set it in `.env` (then restart the app).
- **Notion source shows no pages** — you haven't shared any pages with your
  integration yet (see step 3 above).
- **Answer says "I don't have that information"** — the docs don't cover it;
  add a relevant doc or switch sources, then re-index.
- **Changed a doc but answers are stale** — click **🔄 Re-index** in the sidebar.

---

Built as a learning project — the code favors clarity over cleverness, so you can
read every line and understand exactly how RAG works.
