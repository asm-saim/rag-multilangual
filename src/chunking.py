# chunking.py
import os
import re
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

nltk.download('punkt')

def split_bengali_sentences(text):
    """Custom Bengali sentence splitting using regex and newline handling"""
    # Break on Bengali full stops and punctuation
    sentences = re.split(r"[।!?]", text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, chunk_size=5, overlap=2):
    """Create chunks with overlapping context"""
    sentences = split_bengali_sentences(text)
    chunks = []
    start = 0

    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunk = " ".join(sentences[start:end])

        if chunk.strip() and len(chunk) > 20:
            chunks.append(chunk)

        if end == len(sentences):
            break

        start += (chunk_size - overlap)

    return chunks

if __name__ == "__main__":
    with open("data/cleaned_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    print(f"✅ Chunking done. Total chunks: {len(chunks)}")
