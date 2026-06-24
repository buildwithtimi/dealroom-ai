# test_agent.py
import os
import asyncio
import importlib
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent

# 🟢 Imports the modern, non-deprecated Google GenAI unified SDK classes
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# 1. Align global embedding infrastructure with our cache using the new SDK
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

# 2. Dynamically load our custom workflow tool wrapper
tools_module = importlib.import_module("backend.app.engine.tools.tools")
get_dealroom_workflow_tool = tools_module.get_dealroom_workflow_tool

async def main():
    if not os.path.exists(".chroma_cache"):
        print("❌ Error: Local index cache '.chroma_cache' not found. Run test_ingest.py first!")
        return

    print("💾 Loading cached Vector Index...")
    storage_context = StorageContext.from_defaults(persist_dir=".chroma_cache")
    index = load_index_from_storage(storage_context)

    print("🛠️ Extracting DealRoom workflow tool schema...")
    dealroom_tool = get_dealroom_workflow_tool(index)
    tool_list = [dealroom_tool]

    # 3. 🟢 Initialize the Orchestration Agent using the updated LLM class
    agent_llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

    print("🤖 Initializing Workflows-backed FunctionAgent...")
    agent = FunctionAgent(
        tools=tool_list,
        llm=agent_llm,
        system_prompt=(
            "You are an elite financial due-diligence agent. Use your available tools "
            "intelligently to discover precise company audit metrics."
        )
    )

    agent_query = (
        "Please look into the corporate files using your available tools. "
        "Find out what the current burn rate is and summarize what they are transforming."
    )
    
    print(f"\n🚀 Sending query to Agent: '{agent_query}'\n")
    
    response = await agent.run(user_msg=agent_query)
    
    print("\n✨ Agent's Final Consolidated Answer:")
    print("--------------------------------------")
    print(response)
    print("--------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())