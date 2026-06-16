"""
Notion Knowledge Base Chatbot — a minimal Retrieval-Augmented Generation (RAG) demo.

WHAT IS RAG, IN ONE PARAGRAPH?
An LLM only "knows" what was in its training data. If you want it to answer
questions about YOUR documents (a company wiki, a Notion knowledge base, etc.),
you have two options: fine-tune the model (expensive, slow) or *retrieve* the
relevant text at question time and paste it into the prompt. The second approach
is RAG. The flow is: turn documents into searchable vectors -> when a question
arrives, find the most similar document chunks -> hand those chunks to the LLM
and ask it to answer using only that context. This file implements that flow
from scratch with nothing but `openai` and `numpy`, so every step is visible.

Pipeline (each step is its own function below):
    Load  -> Chunk -> Embed -> Store -> Retrieve -> Generate

Run it with:
    export OPENAI_API_KEY="sk-..."
    python notion_kb_chatbot.py
"""

import os
import glob

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keeping the "knobs" in one place makes the code easy to reason about and tune.

DOCS_DIR = "docs"                       # folder of .txt files to learn from
EMBED_MODEL = "text-embedding-3-small"  # cheap, fast, good-enough embeddings
CHAT_MODEL = "gpt-4o-mini"              # cheap, fast generation model
CHUNK_SIZE = 800                        # characters per chunk (see chunk_text)
CHUNK_OVERLAP = 150                     # characters shared between neighbours
TOP_K = 4                               # how many chunks to retrieve per question

# The OpenAI client reads OPENAI_API_KEY from the environment automatically.
# We construct it once and reuse it everywhere.
client = OpenAI()


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_documents(docs_dir=DOCS_DIR):
    """Read every .txt file in `docs_dir`.

    WHY: RAG can only answer from documents it has seen, so the first job is
    simply to get the raw text into memory. We deliberately keep the filename
    (e.g. "databases.txt") alongside the text — that becomes the "source" we
    cite later, which is what makes RAG answers trustworthy and checkable.

    Returns a list of {"source": <filename>, "text": <file contents>} dicts.
    """
    documents = []
    # sorted() just makes the load order deterministic, which is nice for demos.
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": os.path.basename(path), "text": text})

    if not documents:
        raise SystemExit(f"No .txt files found in '{docs_dir}/'. Add some docs first.")
    return documents


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one document into overlapping, fixed-size character windows.

    WHY CHUNK AT ALL? Two reasons:
      1. Precision — retrieving a small, focused passage gives the LLM exactly
         the relevant sentences instead of a whole document full of noise.
      2. Limits — both embedding models and chat prompts have size limits; we
         cannot embed or paste an entire knowledge base at once.

    THE CHUNK-SIZE TRADEOFF:
      - Chunks too LARGE  -> each vector mixes many topics, so similarity search
        becomes blurry and you waste prompt space on irrelevant text.
      - Chunks too SMALL  -> a single idea gets split across chunks and loses the
        surrounding context needed to make sense of it.
      ~800 characters (a few sentences / a short paragraph) is a good middle
      ground for prose like help docs.

    THE OVERLAP: a sentence that explains a concept might land right on a chunk
    boundary and get cut in half. By letting consecutive chunks share ~150
    characters, an idea that straddles a boundary still appears intact in at
    least one chunk. The price is a little duplicated text.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:                       # skip windows that are only whitespace
            chunks.append(chunk)
        # Advance by (chunk_size - overlap) so the next window backs up a bit
        # and re-includes the tail of this one. This is what creates the overlap.
        start += chunk_size - overlap
    return chunks


def build_chunks(documents):
    """Turn the list of documents into a flat list of chunk records.

    Each record keeps its source filename so we can cite it after retrieval.
    Returns a list of {"source": ..., "text": ...} dicts (one per chunk).
    """
    chunk_records = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunk_records.append({"source": doc["source"], "text": chunk})
    return chunk_records


# ---------------------------------------------------------------------------
# 3. EMBED
# ---------------------------------------------------------------------------
def embed_texts(texts):
    """Convert a list of strings into a NumPy matrix of normalized vectors.

    WHAT IS AN EMBEDDING? A model that maps text to a fixed-length vector of
    numbers such that texts with similar *meaning* end up close together in
    that vector space. This is what lets us search by meaning rather than by
    exact keyword match.

    BATCHING: we send all texts in ONE API call. Embedding is the slow/costly
    part of indexing, and one batched request is far faster and cheaper than N
    separate ones (less network overhead, fewer rate-limit headaches).

    NORMALIZING: cosine similarity measures the *angle* between two vectors,
    ignoring their length. If we scale every vector to length 1 up front, then
    a plain dot product equals cosine similarity. That lets retrieval later be a
    single fast matrix multiply instead of a per-row division. Comparing by
    angle (not magnitude) is what we want, because we care about direction of
    meaning, not how "big" a passage's vector happens to be.

    Returns a (num_texts, embedding_dim) float32 NumPy array.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # response.data preserves input order, so vectors line up with `texts`.
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)

    # Normalize each row to unit length. keepdims lets the division broadcast
    # cleanly across the matrix. (+1e-10 guards against a divide-by-zero.)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    return vectors


# ---------------------------------------------------------------------------
# 4. STORE  +  5. RETRIEVE
# ---------------------------------------------------------------------------
# We bundle storage and retrieval into one small class because they share the
# same data (the vectors and their chunk metadata).

class VectorStore:
    """An in-memory vector index.

    WHY NO DATABASE? A dedicated vector database (Pinecone, FAISS, pgvector...)
    earns its keep when you have hundreds of thousands of vectors, need
    persistence, or want approximate nearest-neighbour search for speed. Here we
    have a few dozen chunks. A plain NumPy array in RAM holds them, and a single
    matrix multiply scores ALL of them in microseconds. Reaching for a database
    at this scale would add complexity, dependencies, and infra for zero benefit.
    The "search engine" is literally one line of NumPy (see retrieve()).
    """

    def __init__(self, chunk_records, vectors):
        self.chunks = chunk_records   # parallel list: metadata for each vector
        self.vectors = vectors        # (num_chunks, dim) normalized matrix

    def retrieve(self, query, top_k=TOP_K):
        """Find the `top_k` chunks most similar to `query`.

        Steps:
          1. Embed the question with the SAME model used for the documents, so
             question and chunks live in the same vector space and are comparable.
          2. Score every chunk by cosine similarity. Because all vectors are
             normalized, that is just a dot product: matrix @ query_vector.
          3. Return the highest-scoring chunks along with their scores.
        """
        query_vector = embed_texts([query])[0]            # shape: (dim,)

        # One matrix-vector product gives a similarity score for EVERY chunk.
        # This is the heart of retrieval — and it's a single NumPy line.
        scores = self.vectors @ query_vector              # shape: (num_chunks,)

        # argsort is ascending; take the last top_k and reverse for descending.
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx]),
            })
        return results


# ---------------------------------------------------------------------------
# 6. GENERATE
# ---------------------------------------------------------------------------
def generate_answer(query, retrieved_chunks, history=None):
    """Ask the LLM to answer the question using only the retrieved chunks.

    This is the "G" in RAG. The retrieval step found relevant text; now we paste
    that text into the prompt as "context" and instruct the model to ground its
    answer in it. The system prompt is doing the heavy lifting here:

      - "answer only from the context" reduces hallucination — the model is told
        not to fall back on its own training-data guesses.
      - "cite the source filename in brackets" makes every claim checkable, which
        is the whole point of a knowledge-base bot.
      - "say you don't have the info" gives the model an honest escape hatch
        instead of forcing it to invent an answer when retrieval came up empty.

    MULTI-TURN: `history` is an optional list of prior {"role", "content"}
    messages (everything before the current question). Passing it lets the model
    resolve follow-ups like "what about sharing it?" against the conversation so
    far. The CLI leaves it None (single-turn); the web app supplies it.

    LOW TEMPERATURE: temperature controls randomness. For factual Q&A we want
    consistent, faithful answers, not creativity, so we set it near 0.
    """
    # Stitch the retrieved chunks into a single context block, each labelled with
    # its source so the model can cite it accurately.
    context = "\n\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful knowledge-base assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "After each claim, cite the source filename in square brackets, e.g. [databases.txt]. "
        "If the answer is not contained in the context, say you don't have that "
        "information — do not make anything up.\n\n"
        f"Context:\n{context}"
    )

    # System prompt first, then any prior turns, then the current question. The
    # context block is rebuilt fresh each turn from THIS question's retrieval,
    # so the model always grounds on the most relevant chunks.
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,  # low = factual and repeatable
        messages=messages,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# main — build the index once, then loop on user questions
# ---------------------------------------------------------------------------
def build_index(documents=None):
    """Run the one-time indexing half of the pipeline: Load -> Chunk -> Embed -> Store.

    `documents` lets a caller supply records from somewhere other than the local
    `docs/` folder (e.g. pulled live from the Notion API) — as long as they're
    the same {"source", "text"} shape, the rest of the pipeline is identical.
    When omitted, we fall back to reading the local .txt files.
    """
    if documents is None:
        print("Loading documents...")
        documents = load_documents()

    print("Chunking...")
    chunk_records = build_chunks(documents)

    print(f"Embedding {len(chunk_records)} chunks in one batched call...")
    vectors = embed_texts([c["text"] for c in chunk_records])

    print(f"Index ready: {len(documents)} docs -> {len(chunk_records)} chunks.\n")
    return VectorStore(chunk_records, vectors)


def main():
    # Fail early with a clear message if the key is missing — a common gotcha.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

    # Indexing is done ONCE up front. The vectors then live in memory and are
    # reused for every question, so each question only pays for one small
    # embedding call + one chat call.
    store = build_index()

    print("Notion KB Chatbot ready. Ask a question, or type 'quit' to exit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        # Retrieve -> Generate, the per-question half of the pipeline.
        retrieved = store.retrieve(query)
        answer = generate_answer(query, retrieved)

        print(f"\nBot: {answer}")

        # Show the retrieval internals so you can SEE the RAG working: which
        # chunks were pulled and how similar each was to the question (1.0 = a
        # perfect match). This transparency is great for learning and debugging.
        print("\n--- retrieved chunks (source : similarity) ---")
        for chunk in retrieved:
            preview = chunk["text"][:70].replace("\n", " ")
            print(f"  {chunk['source']:<24} {chunk['score']:.3f}  | {preview}...")


if __name__ == "__main__":
    main()
