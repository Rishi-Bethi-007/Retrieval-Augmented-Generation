from typing import List

def chunk(text: str, chunk_size:int =400) -> List[str]:
    """
    Fixed-size chunking by characters.

    - Simple and fast
    - Can split sentences/words (lower quality)
    """
    text = text.strip() # Remove leading/trailing whitespace
    if not text:
        return [] # Return empty list for empty input
    
    chunks: List[str] = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks