# Semantic Search Engine (FAISS + Sentence Transformers)

This project implements a **semantic search engine from scratch**, focusing on retrieval quality, system design, and evaluation.  
The system retrieves relevant text chunks based on **semantic similarity**, not keyword matching, using dense embeddings and a FAISS vector index.

This project is intentionally built **without any LLMs** to deeply understand retrieval mechanics before introducing Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

Keyword-based search often fails when queries and documents use different wording for the same concept.  
This project solves that problem by:

- converting text into dense vector embeddings
- indexing them in a vector database (FAISS)
- retrieving the most semantically relevant chunks for a given query

The result is a **retrieval backbone** that can later be reused directly in RAG systems.

---

## 🧠 System Architecture

### Offline Indexing Pipeline
# Semantic Search Engine (FAISS + Sentence Transformers)

This project implements a **semantic search engine from scratch**, focusing on retrieval quality, system design, and evaluation.  
The system retrieves relevant text chunks based on **semantic similarity**, not keyword matching, using dense embeddings and a FAISS vector index.

This project is intentionally built **without any LLMs** to deeply understand retrieval mechanics before introducing Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

Keyword-based search often fails when queries and documents use different wording for the same concept.  
This project solves that problem by:

- converting text into dense vector embeddings
- indexing them in a vector database (FAISS)
- retrieving the most semantically relevant chunks for a given query

The result is a **retrieval backbone** that can later be reused directly in RAG systems.

---

## 🧠 System Architecture

### Offline Indexing Pipeline
Documents (.txt / .md)
↓
Chunking (sentence-aware)
↓
Embedding (sentence-transformers)
↓
Mean pooling + L2 normalization
↓
FAISS vector index + JSON metadata

### Online Query Pipeline

---

## ✂️ Chunking Strategies

The system supports multiple chunking strategies to study their impact on retrieval quality:

- **Fixed-size chunking** (character-based)
- **Overlapping chunking** (character-based with overlap)
- **Sentence-aware chunking (default)**  
  Preserves semantic boundaries and improves retrieval quality for conceptual queries.

Chunking is treated as a **first-class design decision**, not a preprocessing detail.

---

## 🔢 Embeddings & Similarity

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Pooling**: Mean pooling over token embeddings
- **Normalization**: L2 normalization
- **Similarity metric**: Cosine similarity (via inner product)

Normalization ensures:
- stable similarity scores
- fair comparison across different chunk lengths
- correct behavior with FAISS `IndexFlatIP`

---

## 🗄️ Vector Index

- **Library**: FAISS
- **Index type**: `IndexFlatIP` (exact search)

**Why exact search?**
- correctness over scale at this stage
- easier debugging and evaluation
- clean baseline before approximate methods (IVF / HNSW)

The index and metadata are persisted to disk and reloaded for querying.

---

## 🔍 Usage
### Build the index
Place `.txt` or `.md` files in `data/docs/`, then run:
```bash
python -m index.build_index

This creates:

index/chunk_index.faiss

index/chunk_metadata.json

Search

python -m index.search "What is a vector database?" --top_k 5


📊 Evaluation & Key Insight

The system was evaluated using a gold query set, mapping queries to expected documents.

Metric used

Doc Hit@3 (whether the correct document appears in the top-3 retrieved documents)

Key Observation

“I evaluated my semantic search system with a gold query set and observed that abstract conceptual queries often require LLM-based synthesis beyond pure embedding retrieval.”

Interpretation

Concrete, well-scoped queries (e.g. “What is FAISS used for?”) are retrieved accurately.

Abstract or explanatory queries (e.g. “Explain cosine similarity”) often retrieve semantically related documents rather than a single definitive source.

This behavior is expected for embedding-only retrieval and directly motivates the use of LLMs for synthesis in RAG systems.

🧪 Evaluation Script
python -m evaluation.eval_retrieval --top_k_docs 3


Outputs:

HIT / MISS per query

overall Doc Hit@K score

🧩 Design Decisions

Retrieval-first approach before introducing LLMs

Explicit separation of:

ingestion

chunking

embedding

indexing

querying

Offline indexing vs online querying

Human-readable metadata for debuggability

This mirrors how real-world AI systems are designed and debugged.

📦 Project Structure
semantic-search/
├── data/docs/              # raw text documents
├── chunking/               # chunking strategies
├── embeddings/             # embedding logic
├── index/                  # index build + search
├── evaluation/             # evaluation scripts
├── config.py
├── requirements.txt
└── README.md
