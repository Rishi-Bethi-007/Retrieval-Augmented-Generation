import json
from index.search import FaissRetriever

def main():
    retriever = FaissRetriever(
        index_path="index/chunk_index.faiss",
        meta_path="index/chunk_metadata.json",
    )

    gold = json.load(open("evaluation/gold_rag_eval.json", "r", encoding="utf-8"))

    hit3 = 0
    total = 0

    for g in gold:
        q = g["query"]
        expected = g["expected_doc"]

        chunks = retriever.retrieve(q, top_k=8)

        # unique docs in retrieved order
        top_docs = []
        for c in chunks:
            if c["doc_id"] not in top_docs:
                top_docs.append(c["doc_id"])

        top3 = top_docs[:3]

        total += 1
        ok = (expected in top3) if expected else (len(top3) == 0)
        hit3 += int(ok)

        print("\nQuery:", q)
        print("Expected:", expected)
        print("Top-3 docs:", top3)
        print("✅ HIT" if ok else "❌ MISS")

    print("\n=== SUMMARY ===")
    print(f"Doc Hit@3: {hit3}/{total} = {hit3/total:.2%}")

if __name__ == "__main__":
    main()
