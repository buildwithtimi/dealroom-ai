# backend/app/engine/01_ingestion/indexer.py
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

class DealRoomIndexer:
    def __init__(self, cache_dir: str = ".chroma_cache"):
        self.cache_dir = cache_dir
        
        # 🟢 Explicitly switch from the deprecated text-embedding-004 
        # to the active, production-grade gemini-embedding-001 model
        self.embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001"
        )
        
        # Assign it globally to LlamaIndex settings
        Settings.embed_model = self.embed_model

    def build_vector_store(self, documents):
        if not documents:
            print("⚠️ No documents passed to indexer.")
            return None
            
        print("🧠 Chunking text and generating Gemini embeddings...")
        
        # Build the index using our explicitly defined active model
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist locally to prevent burning API quota on future runs
        index.storage_context.persist(persist_dir=self.cache_dir)
        print(f"💾 Vector index successfully saved locally to '{self.cache_dir}'")
        return index