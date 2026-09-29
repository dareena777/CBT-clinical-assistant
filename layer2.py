from __future__ import annotations
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional



class RAGLayer2:
    def __init__(self, db_path: str = "./chroma_db"):
        # Shared embedding model for tokenization & vectorization
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="cbt_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------------------
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """Splits text into overlapping character chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def ingest_json(self, json_path: str = "./cbt_knowledge.json"):
        """Loads JSON/JSONL, splits into chunks, tokenizes, embeds, and stores in ChromaDB."""
        if not os.path.exists(json_path):
            print(f"Error: Knowledge base file '{json_path}' not found.")
            return

        docs, metadatas = [], []

        with open(json_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]

        for item in data:
            full_text = f"Title: {item.get('title', '')}\nContent: {item.get('content', '')}\nInstructions: {item.get('instructions', '')}"
            for idx, chunk in enumerate(self.chunk_text(full_text)):
                docs.append(chunk)
                metadatas.append({
                    "title": item.get("title", ""),
                    "category": item.get("category", "general").lower(),
                    "target_condition": item.get("target_condition", "general").lower(),
                    "chunk_index": idx
                })

        self.collection.upsert(
            documents=docs,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        print(f"Successfully chunked, embedded, and stored {len(docs)} documents into ChromaDB.")

    # -------------------------------------------------------------------------
    # 2. RETRIEVAL LOGIC
    # -------------------------------------------------------------------------
    def retrieve_context(
        self,
        query: str,
        category: Optional[str] = None,
        target_condition: Optional[str] = None,
        top_k: int = 2
    ) -> list[str]:
        """
        Embeds the query using the same model, applies optional Layer 1 metadata filtering,
        and retrieves top-k text chunks.
        """
        where_clause = {}
        if category and target_condition:
            where_clause = {
                "$and": [
                    {"category": category.lower()},
                    {"target_condition": target_condition.lower()}
                ]
            }
        elif category:
            where_clause = {"category": category.lower()}
        elif target_condition:
            where_clause = {"target_condition": target_condition.lower()}

        # ChromaDB vectorizes query text using self.embedding_fn automatically
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause if where_clause else None
        )

        if results and results.get("documents") and results["documents"][0]:
            return results["documents"][0]
        return []


if __name__ == "__main__":
    rag = RAGLayer2()
    
    # 1. Ingest Data
    rag.ingest_json("./cbt_knowledge.json")
    
    # 2. Test Retrieval with Layer 1 Metadata Signals
    sample_query = "My mind won't stop racing and I feel overwhelmed."
    retrieved_chunks = rag.retrieve_context(
        query=sample_query,
        category="anxiety",
        target_condition="stress",
        top_k=2
    )
    
    print("\n=== RETRIEVED CHUNKS ===")
    for idx, chunk in enumerate(retrieved_chunks, 1):
        print(f"[{idx}] {chunk}\n")