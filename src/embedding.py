# embedding.py
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
chroma_path = "data/chroma_db"

# Clear existing database
if os.path.exists(chroma_path):
    import shutil
    shutil.rmtree(chroma_path)

with open("data/chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines() if line.strip()]

print(f"⚙️ Generating embeddings for {len(chunks)} chunks...")
embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(name="rag_collection")

print("📦 Storing embeddings in ChromaDB...")
batch_size = 100
for i in tqdm(range(0, len(chunks), batch_size), desc="Storing"):
    batch_chunks = chunks[i:i+batch_size]
    batch_embeddings = embeddings[i:i+batch_size].tolist()
    batch_ids = [str(j) for j in range(i, min(i+batch_size, len(chunks)))]
    
    collection.add(
        documents=batch_chunks,
        embeddings=batch_embeddings,
        ids=batch_ids
    )

print(f"✅ {len(chunks)} embeddings stored in ChromaDB.")