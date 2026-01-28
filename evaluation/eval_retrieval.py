import json
import argparse

import faiss

from embeddings.embedder import Embedder
from index.search import search, rank_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_path", type=str, default="evaluation/queries.json")
    parser.add_argument("--index_path", type=str, default="index/chunk_index.faiss")
    parser.add_argument("--meta_path", type=str, default="index/chunk_metadata.json")
    parser.add_argument("--top_k_chunks", type=int, default=10)
    parser.add_argument("--top_k_docs", type=int, default=3)
    args = parser.parse_args()

    with open(args.queries_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    index = faiss.read_index(args.index_path)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(f"Index vectors ({index.ntotal}) != metadata rows ({len(metadata)})")

    embedder = Embedder()

    hits = 0
    for t in tests:
        query = t["query"]
        expected = t["expected_doc"]

        chunk_results = search(
            query=query,
            embedder=embedder,
            index=index,
            metadata=metadata,
            top_k=args.top_k_chunks,
        )
        doc_ranks = rank_documents(chunk_results)

        top_docs = [d["doc_id"] for d in doc_ranks[: args.top_k_docs]]
        ok = expected in top_docs

        hits += int(ok)
        status = "✅ HIT" if ok else "❌ MISS"

        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Top-{args.top_k_docs} docs: {top_docs}")
        print(status)

    total = len(tests)
    print("\n=== SUMMARY ===")
    print(f"Doc Hit@{args.top_k_docs}: {hits}/{total} = {hits/total:.2%}")


if __name__ == "__main__":
    main()
