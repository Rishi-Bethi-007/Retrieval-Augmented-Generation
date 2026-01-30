import json
from rag.rag_pipeline import run_rag
from index.search import FaissRetriever

def main():
    retriever = FaissRetriever(
        index_path="index/chunk_index.faiss",
        meta_path="index/chunk_metadata.json",
    )

    gold = json.load(open("evaluation/gold_rag_eval.json", "r", encoding="utf-8"))

    refusal_correct = 0
    citation_valid = 0
    total = 0

    for g in gold:
        q = g["query"]
        answerable = g["answerable"]

        # retrieve context (what we *gave* the model)
        chunks = retriever.retrieve(q, top_k=5)

        # set of allowed citations from retrieved context
        allowed = set((c["doc_id"], c["chunk_id"]) for c in chunks)

        out = run_rag(q, top_k=5)
        total += 1

        is_refusal = out.answer.strip().lower() == "i don't know"
        if (not answerable and is_refusal) or (answerable and not is_refusal):
            refusal_correct += 1

        # citations should be subset of retrieved chunks
        ok_cite = True
        for cit in out.citations:
            if (cit.doc_id, cit.chunk_id) not in allowed:
                ok_cite = False
                break
        citation_valid += int(ok_cite)

        print("\nQuery:", q)
        print("Answer:", out.answer)
        print("Confidence:", out.confidence)
        print("Citations:", [(c.doc_id, c.chunk_id) for c in out.citations])
        print("Refusal OK:", "✅" if ((not answerable and is_refusal) or (answerable and not is_refusal)) else "❌")
        print("Citations OK:", "✅" if ok_cite else "❌")

    print("\n=== SUMMARY ===")
    print(f"Refusal correctness: {refusal_correct}/{total} = {refusal_correct/total:.2%}")
    print(f"Citation validity:   {citation_valid}/{total} = {citation_valid/total:.2%}")

if __name__ == "__main__":
    main()
