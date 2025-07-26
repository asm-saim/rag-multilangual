import os
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb import PersistentClient
import ollama
import re

# ChromaDB setup
chroma_path = "data/chroma_db"
client = PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(name="rag_collection")

# Embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Expanded Bengali synonyms for query expansion
BENGALI_SYNONYMS = {
    "বয়স": ["বয়স", "উমর", "সাল", "আয়ু", "বয়স্কতা"],
    "প্রকৃত": ["প্রকৃত", "আসল", "বাস্তব", "সত্যিকারের", "মূল"],
    "বিয়ে": ["বিবাহ", "শাদী", "পরিণয়", "বিয়েবাড়ি", "বনধন"],
    "সুপুরুষ": ["সুপুরুষ", "সুন্দর পুরুষ", "শুম্ভুনাথ", "আকর্ষণীয় পুরুষ"],
    "ভাগ্য": ["ভাগ্য", "নিয়তি", "কপাল", "ভাগ্যদেবতা", "অদৃষ্ট"]
}

def expand_query(query):
    """Expand query with Bengali synonyms for better retrieval"""
    expanded = [query]
    for word, synonyms in BENGALI_SYNONYMS.items():
        if word in query:
            for syn in synonyms:
                new_query = query.replace(word, syn)
                if new_query != query:
                    expanded.append(new_query)
    return expanded

def retrieve_context(query: str, top_k=5) -> str:
    """Enhanced retrieval with query expansion and reranking"""
    queries = expand_query(query)
    query_embeddings = embedding_model.encode(queries).tolist()
    
    # Retrieve more candidates initially
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k * 3  # Get extra for reranking
    )
    
    # Rerank by semantic similarity to original query
    original_embedding = embedding_model.encode(query)
    documents = [doc for sublist in results["documents"] for doc in sublist]
    doc_embeddings = embedding_model.encode(documents)
    
    # Calculate cosine similarities
    cos_scores = util.pytorch_cos_sim(original_embedding, doc_embeddings)[0]
    
    # Combine scores and documents
    scored_docs = list(zip(cos_scores.tolist(), documents))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for score, doc in scored_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append((score, doc))
    
    # Sort by similarity score
    unique_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select top results
    top_docs = [doc for _, doc in unique_docs[:top_k]]
    return "\n\n".join(f"[উদ্ধৃতি {i+1}] {doc}" for i, docile_doc in enumerate(top_docs))

def build_prompt(query: str, context: str) -> str:
    """Improved prompt structure for factual responses"""
    return f"""
# নির্দেশাবলী:
1. নিচের প্রাসঙ্গিক উদ্ধৃতিগুলো ব্যবহার করে প্রশ্নের উত্তর দিন
2. শুধুমাত্র প্রদত্ত প্রাসঙ্গিক তথ্যের উপর ভিত্তি করে উত্তর দিন
3. উত্তরটি সংক্ষিপ্ত এবং সুনির্দিষ্ট রাখুন (একটি শব্দ বা ছোট বাক্য)
4. উত্তর অবশ্যই বাংলায় দিতে হবে
5. যদি উত্তর প্রাসঙ্গিক উদ্ধৃতিতে না থাকে, "উত্তর পাওয়া যায়নি" বলুন

# প্রাসঙ্গিক উদ্ধৃতিঃ
{context}

# প্রশ্নঃ
{query}

# উত্তরের ফরম্যাট:
উত্তর: [এখানে আপনার সংক্ষিপ্ত উত্তর দিন]
"""

def validate_answer(answer, context):
    """Enhanced validation using semantic similarity"""
    if not answer or not context:
        return False
    answer_embedding = embedding_model.encode(answer)
    context_embedding = embedding_model.encode(context)
    similarity = util.pytorch_cos_sim(answer_embedding, context_embedding)[0][0]
    return similarity > 0.7  # Threshold for semantic similarity

def ask_question(query: str):
    """Enhanced RAG flow with validation"""
    context = retrieve_context(query, top_k=5)
    
    print(f"\n📌 প্রশ্ন: {query}")
    print(f"\n📚 প্রাসঙ্গিক অংশ:\n{context}\n")
    
    if not context.strip():
        print("⚠️ No relevant context found.")
        return "উত্তর পাওয়া যায়নি"
    
    prompt = build_prompt(query, context)
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    
    answer = response["message"]["content"].strip()
    
    # Extract answer from structured response
    if "উত্তর:" in answer:
        answer = answer.split("উত্তর:")[-1].strip()
    
    # Validate against context
    if not validate_answer(answer, context):
        answer = "উত্তর পাওয়া যায়নি"
    
    print(f"\n🧠 উত্তর:\n{answer}")
    return answer

if __name__ == "__main__":
    questions = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    for q in questions:
        ask_question(q)