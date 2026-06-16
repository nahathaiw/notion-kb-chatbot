# System Architecture

A flowchart-based overview of the Notion KB Chatbot. Diagrams use
[Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

---

## 1. High-level overview

How the pieces fit together: a Streamlit UI on top of a from-scratch RAG engine,
fed by either local files or a live Notion workspace, talking to the OpenAI API.

```mermaid
flowchart TD
    User([User]) -->|asks question| UI[app.py<br/>Streamlit chat UI]

    subgraph Sources [Knowledge sources]
        Docs[/"docs/*.txt"/]
        Notion[(Notion workspace)]
    end

    Docs -->|load_documents| RAG
    Notion -->|notion_loader.py<br/>fetch_notion_documents| RAG

    UI <-->|retrieve + generate| RAG[notion_kb_chatbot.py<br/>RAG engine]
    RAG <-->|embeddings + chat| OpenAI{{OpenAI API}}

    UI -->|answer + citations + chunks| User

    classDef entry fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
    classDef ext fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    class UI,RAG entry;
    class OpenAI,Notion ext;
```

---

## 2. Indexing pipeline (build once)

Runs once per source and is cached (`st.cache_resource`), so documents are
embedded a single time and reused across every question.

```mermaid
flowchart LR
    A[Load<br/>load_documents /<br/>fetch_notion_documents] --> B[Chunk<br/>chunk_text + build_chunks<br/>~800 chars, 150 overlap]
    B --> C[Embed<br/>embed_texts<br/>one batched API call]
    C --> D[Normalize<br/>unit vectors]
    D --> E[(Store<br/>VectorStore<br/>NumPy matrix in RAM)]

    classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d;
    class A,B,C,D step;
```

Each chunk keeps its `source` (filename or Notion page title) so answers can
cite it.

---

## 3. Query → answer flow (per question)

The per-question path, including the two quality features: **query rewriting**
for multi-turn context and the **relevance threshold** that lets the bot say "I
don't know" honestly.

```mermaid
flowchart TD
    Q([User question]) --> H{Has prior<br/>conversation?}
    H -->|yes| RW[rewrite_query<br/>follow-up → standalone query]
    H -->|no| SkipRW[use question as-is]
    RW --> EM
    SkipRW --> EM

    EM[Embed query<br/>embed_texts] --> SC[Score all chunks<br/>vectors · query<br/>cosine similarity]
    SC --> TK[Take top-k]
    TK --> TH{Any chunk ≥<br/>MIN_SIMILARITY?}

    TH -->|no| NoInfo[Return honest<br/>'I don't have that info'<br/>no API call]
    TH -->|yes| GEN[generate_answer<br/>chunks + history + question<br/>→ GPT, temperature 0.1]

    GEN --> Ans([Answer with<br/>source citations])
    NoInfo --> Ans

    Ans --> Panel[Show retrieved chunks<br/>+ similarity scores]

    classDef dec fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    classDef act fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
    class H,TH dec;
    class RW,EM,SC,TK,GEN,NoInfo act;
```

---

## 4. Source switching & caching

How the app decides what to index and when to rebuild.

```mermaid
flowchart TD
    Side[Sidebar:<br/>Knowledge source] --> Choice{Source?}

    Choice -->|Local docs| Sig[fingerprint docs/<br/>filenames + mtimes]
    Choice -->|Notion workspace| NSig[key = 'notion']

    Sig --> Cache
    NSig --> Cache

    Cache{In st.cache_resource?}
    Cache -->|hit| Reuse[Reuse VectorStore]
    Cache -->|miss / changed| Build[build_index<br/>→ re-embed]

    Reuse --> Ready([Index ready])
    Build --> Ready

    ReIdx[🔄 Re-index button] -->|get_store.clear| Build

    classDef dec fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    class Choice,Cache dec;
```

The local-docs fingerprint means editing or adding a `.txt` auto-rebuilds the
index. Notion content can't be cheaply fingerprinted, so it's refreshed manually
via **Re-index**.

---

## 5. Request lifecycle (startup → first answer)

```mermaid
sequenceDiagram
    actor U as User
    participant A as app.py
    participant R as RAG engine
    participant N as notion_loader
    participant O as OpenAI API

    U->>A: open app / enter password
    A->>A: load .env, check APP_PASSWORD
    alt Notion source
        A->>N: fetch_notion_documents()
        N->>O: (none — uses Notion API)
    end
    A->>R: build_index(documents)
    R->>O: embed all chunks (batched)
    O-->>R: vectors
    R-->>A: VectorStore (cached)

    U->>A: ask question
    A->>R: rewrite_query(q, history)
    R->>O: chat (rewrite)
    O-->>R: standalone query
    A->>R: retrieve(query, top_k, min_score)
    R->>O: embed query
    O-->>R: top chunks (≥ threshold)
    A->>R: generate_answer(q, chunks, history)
    R->>O: chat (grounded prompt)
    O-->>R: answer + citations
    R-->>A: answer
    A-->>U: answer + retrieved chunks
```

---

## Component reference

| Component | File | Responsibility |
|-----------|------|----------------|
| Web UI | `app.py` | Chat interface, password gate, source switch, index caching, displays citations & retrieved chunks |
| RAG engine | `notion_kb_chatbot.py` | Load, chunk, embed, store, retrieve, rewrite_query, generate_answer; also runs as a CLI |
| Notion source | `notion_loader.py` | Fetch pages via the Notion API and extract text |
| Knowledge base | `docs/*.txt` | Local source documents |
| Config | `.streamlit/config.toml` | Port 8502, headless |
| Secrets | `.env` | `OPENAI_API_KEY`, `NOTION_API_KEY`, `APP_PASSWORD` (gitignored) |

**Models:** embeddings `text-embedding-3-small`, generation `gpt-4o-mini`.
