
# Multilingual RAG System

## Features
- English + Bangla Query
- PDF Knowledge Base
- Embedding + Retrieval + Generation
- API Access

## Tools Used
- PyMuPDF
- NLTK
- Sentence Transformers
- ChromaDB
- Ollama + LLaMA3
- FastAPI

## Setup
1. Clone repo
2. Install requirements: `pip install -r requirements.txt`
3. Run preprocessing, chunking, and embedding.
4. Launch API or use CLI.

## Sample Query
POST `/ask`
```json
{ "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" }
```
