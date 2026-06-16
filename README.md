# Notion Knowledge Base Chatbot (minimal RAG demo)

A tiny, dependency-light **Retrieval-Augmented Generation (RAG)** chatbot that
answers questions about a folder of `.txt` documents — built from scratch with
only `openai` and `numpy` so you can read and explain every line.

## What it does

It indexes the docs in `docs/` once, then lets you ask questions in a loop. For
each question it retrieves the most relevant document chunks and asks GPT to
answer **using only those chunks**, citing the source filename for every claim.

The RAG pipeline, each as a clearly separated function in `notion_kb_chatbot.py`:

| Step | Function | What it does |
|------|----------|--------------|
| Load     | `load_documents` | Read every `.txt` file from `docs/`, keeping the filename as the citation source |
| Chunk    | `chunk_text` / `build_chunks` | Split docs into overlapping ~800-char windows |
| Embed    | `embed_texts` | Convert chunks to normalized vectors in one batched API call |
| Store    | `VectorStore` | Hold the vectors in memory as a NumPy array |
| Retrieve | `VectorStore.retrieve` | Embed the question, score all chunks by cosine similarity, return top-k |
| Generate | `generate_answer` | Pass the chunks to GPT with a grounded, citation-required system prompt |

## Setup

Requires Python 3.8+.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # your OpenAI API key
```

## Run

Two front-ends share the same RAG pipeline.

**Web app (Streamlit)** — a chat UI with conversation memory, a cached index,
and a per-answer panel showing which chunks were retrieved:

```bash
streamlit run app.py
```

**CLI** — the original terminal loop:

```bash
python notion_kb_chatbot.py
```

## Connect your real Notion workspace

The web app can answer from your actual Notion pages instead of the local
`.txt` files. One-time setup:

1. Create an internal integration at <https://www.notion.so/my-integrations>
   and copy its **Internal Integration Secret**.
2. Add it to `.env`:
   ```bash
   NOTION_API_KEY=secret_xxx
   ```
3. Share pages with the integration: open a page → **•••** → **Connections** →
   add your integration. The API only sees pages you explicitly share.
4. In the app sidebar, switch **Knowledge source** to **Notion workspace**.
   Use **🔄 Re-index** to pull the latest pages.

`notion_loader.py` does the fetching — it searches every accessible page, walks
each page's block tree to extract the text, and hands it to the same
chunk → embed → retrieve → generate pipeline.

You'll get an interactive prompt. Try:

- `How do I create a database?`
- `What permission levels does Notion have?`
- `How do I share a page to the web?`
- `What is a block?`

Type `quit` to exit.

After each answer the script prints the retrieved chunks and their similarity
scores (1.0 = perfect match) so you can watch retrieval working.

## Models used

- Embeddings: `text-embedding-3-small`
- Generation: `gpt-4o-mini`

## Try it with your own docs

Drop any `.txt` files into `docs/` and re-run. The filename becomes the citation
label, so name them descriptively.
