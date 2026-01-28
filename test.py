from chunking.fixed import chunk as fixed_chunk
from chunking.overlap import chunk as overlap_chunk
from chunking.sentence_aware import chunk as sent_chunk

text = "LoRA fine-tuning trains low-rank adapters while keeping the base model frozen. It is efficient and modular."

print("FIXED:", fixed_chunk(text, chunk_size=40))
print("OVERLAP:", overlap_chunk(text, chunk_size=40, overlap=10))
print("SENT:", sent_chunk(text, chunk_size=60))
