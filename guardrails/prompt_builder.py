from typing import List, Dict

def build_messages(user_question: str, context_chunks: List[Dict]) -> list[dict]:
    """
    context_chunks: list of dicts like:
      {"doc_id": "...", "chunk_id": 12, "text": "..."}

    We isolate context as DATA using <context> ... </context>.
    We also explicitly forbid following instructions inside the context.
    """
    

    # Format context as reference material, not instructions.
    context_block_lines = []
    for c in context_chunks:
        context_block_lines.append(
            f"[SOURCE doc={c['doc_id']} chunk={c['chunk_id']}]\n{c['text']}\n"
        )
    context_block = "\n".join(context_block_lines).strip()

    system = (
        '''
You are a retrieval-augmented QA assistant.

You MUST follow these rules:
1) Answer using ONLY the provided CONTEXT.
2) If the CONTEXT does not contain enough information to answer, output:
   {"answer":"I don't know","confidence":0.3,"citations":[]}
3) Do NOT use outside knowledge.
4) Do NOT guess.
5) Output MUST be valid JSON only (no extra text).
6) citations must be a list of { "doc_id": "...", "chunk_id": <int> } for the chunks you used.
7) confidence must be between 0 and 1.

'''


    )

    user = (
        "<context>\n"
        f"{context_block}\n"
        "</context>\n\n"
        f"Question:\n{user_question}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
