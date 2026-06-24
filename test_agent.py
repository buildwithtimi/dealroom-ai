# test_agent.py
import os
import asyncio
import importlib
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent

# 🟢 Import the dedicated memory management module
from llama_index.core.memory import ChatMemoryBuffer

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

    # 3. Initialize the Orchestration Agent using the updated LLM class
    agent_llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

    # 4. 🟢 Allocate a token-capped conversational memory buffer
    print("🧠 Initializing persistent Chat Memory Buffer...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    print("🤖 Initializing Workflows-backed FunctionAgent with Memory...")
    agent = FunctionAgent(
        tools=tool_list,
        llm=agent_llm,
        memory=memory,  # 🟢 Hooking the memory engine directly into the agent lifecycle
        system_prompt=(
            "You are an elite financial due-diligence agent. Use your available tools "
            "intelligently to discover precise company audit metrics. Pay close attention "
            "to details in the chat history to resolve context over multi-turn conversations."
        )
    )

    # --- SIMULATED MULTI-TURN DIALOGUE ---

    # 🚀 TURN 1: Explicit instruction requiring the Workflow tool
    query_1 = (
        "Please look into the corporate files using your available tools. "
        "Find out what the current burn rate is and summarize what they are transforming."
    )
    print(f"\n[TURN 1] 💬 User: '{query_1}'")
    print("Thinking...")
    
    response_1 = await agent.run(user_msg=query_1)
    print(f"✨ Agent Response 1:\n{response_1}\n")
    print("-" * 50)

    # 🚀 TURN 2: Contextual follow-up question
    # This completely depends on memory to know what "that burn rate" means.
    query_2 = "Based on that burn rate, if the company has a cash balance of $500,000, how many months of runway do they have left?"
    print(f"\n[TURN 2] 💬 User: '{query_2}'")
    print("Thinking (utilizing historical context)...")
    
    response_2 = await agent.run(user_msg=query_2)
    print(f"✨ Agent Response 2:\n{response_2}\n")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())