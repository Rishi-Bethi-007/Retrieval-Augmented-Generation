import os
import json
from typing import List, Tuple

import faiss
import numpy as np

from embeddings.embedder import Embedder
from chunking.sentence_aware import chunk as sentence_chunker

## Load documents from the disk
def load_documents(docs_dir:str)-> List[Tuple[str, str]]:
    """
    Load all .txt and .md files from a directory.

    Returns:
        List of (doc_id, text)
    """

    documents= []

    # Iterate through files in the directory and load text files
    for fname in sorted(os.listdir(docs_dir)):

        # Skip non-text files
        if not (fname.endswith(".txt") or fname.endswith(".md")):
                continue
               
        path = os.path.join(docs_dir, fname) # Get full file path
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if text:
             doc_id = fname
             documents.append((doc_id, text))

    return documents



## Build chunks + metadata

def build_chunks(documents):
    """
    Build chunks from documents using SentenceAwareChunker.

    Returns:
        List of (chunk_id, chunk_text, metadata)
    """
    chunk_texts = []
    metadata = []
    chunk_index = 0

    for doc_id, text in documents:
        chunks = sentence_chunker(text)

        for chunk_id, ch in enumerate(chunks):
            chunk_texts.append(ch)
            metadata.append({
                'index_id': chunk_index,
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'text': ch                    
            })

            chunk_index += 1

    return chunk_texts, metadata


## Embedd chunks + build FAISS index

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP index from embeddings.

    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


## Main execution function

def main():
     DOCS_DIR ='data/docs'
     INDEX_PATH = 'index/chunk_index.faiss'
     META_PATH = "index/chunk_metadata.json"

     print("Loading documents...")

     documents = load_documents(DOCS_DIR)
     print(f"Loaded {len(documents)} documents.")

     print("chunking documents...")
     chunk_texts, metadata = build_chunks(documents)
     print(f"Built {len(chunk_texts)} chunks.")

     print("Embedding chunks...")
     embedder = Embedder()
     embeddings = embedder.embed(chunk_texts)
     emb_np = embeddings.numpy().astype('float32')

     print("Building FAISS index...")
     index = build_faiss_index(emb_np)

     print("saving index...")
     faiss.write_index(index, INDEX_PATH)

     print('saving metadata...')
     with open(META_PATH, 'w', encoding='utf-8') as f:
         json.dump(metadata, f, ensure_ascii=False, indent=2)

     print("Done.")
     print(f"Index saved to {INDEX_PATH}")
     print(f"Metadata saved to {META_PATH}")

if __name__ == "__main__":
    main()