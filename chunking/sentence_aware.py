from typing import List
import re

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def chunk(text: str, chunk_size: int = 400 ) -> List[str]:
    """
    Sentence-aware chunking.

    - Splits into sentences first
    - Packs sentences into chunks up to chunk_size chars
    - Preserves meaning boundaries better than fixed slicing
    """
    text = text.strip()  # Remove leading/trailing whitespace
    if not text:
        return []  # Return empty list for empty input

    sentences = _SENT_SPLIT.split(text)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences: # Process each sentence
        sentence = sentence.strip()
        if not sentence:
            continue

        #if sentence itself is longer than chunk_size, we need to split it
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # hard split long sentence into chunk_size parts
            for i in range(0, len(sentence), chunk_size):
                sub_chunk = sentence[i:i + chunk_size].strip()
                if sub_chunk:
                    chunks.append(sub_chunk)
            continue

        #If it fits , append to current chunk
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()

        else:
            # current chunk is full; push it and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks