# test_generate.py
import os
import asyncio
import importlib
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

# 1. Setup our active Embedding Model to match our saved index
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

# 2. Dynamically import the workflow and events using importlib
generation_workflow_module = importlib.import_module("backend.app.engine.02_generation.workflow")
DealRoomWorkflow = generation_workflow_module.DealRoomWorkflow

async def main():
    # Verify the cache exists before loading
    if not os.path.exists(".chroma_cache"):
        print("❌ Error: Local index cache '.chroma_cache' not found. Please run test_ingest.py first!")
        return

    print("💾 Loading cached Vector Index...")
    storage_context = StorageContext.from_defaults(persist_dir=".chroma_cache")
    index = load_index_from_storage(storage_context)

    # 3. Initialize the Workflow with our index
    print("🏗️ Initializing Event-Driven Generation Workflow...")
    workflow = DealRoomWorkflow(index=index, timeout=60.0)

    # 4. Trigger the State Machine with a query
    test_query = "What is the burn rate of the company and what are they transforming?"
    print(f"\n🚀 Launching Workflow State Machine with query: '{test_query}'")
    
    result = await workflow.run(query=test_query)
    
    print("\n✨ Final Synthesis Result:")
    print("--------------------------")
    print(result)
    print("--------------------------")

if __name__ == "__main__":
    # Run the async loop smoothly
    asyncio.run(main())