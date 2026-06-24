# backend/app/engine/02_generation/workflow.py
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
import os

from .events import RetrieveEvent

class DealRoomWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.index = index
        
        # Initialize the production-grade Gemini LLM
        self.llm = Gemini(
            model_name="models/gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        Settings.llm = self.llm

    @step
    async def retrieve(self, ev: StartEvent) -> RetrieveEvent:
        """Step 1: Accept query, query vector index, and pass down context nodes."""
        query = ev.get("query")
        if not query:
            raise ValueError("Workflow started without a valid 'query'.")
            
        print(f"🔍 Searching Vector Store for: '{query}'")
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query)
        
        return RetrieveEvent(query=query, nodes=nodes)

    @step
    async def synthesize(self, ev: RetrieveEvent) -> StopEvent:
        """Step 2: Take context nodes, format into a clean prompt, and synthesize answer."""
        context_str = "\n\n".join([node.node.get_content() for node in ev.nodes])
        
        prompt = f"""
        You are an advanced DealRoom AI due diligence assistant. 
        Analyze the provided context and answer the user query professionally.
        
        Context:
        {context_str}
        
        Query: {ev.query}
        Answer:
        """
        
        print("🧠 Synthesizing final analytical response via Gemini...")
        response = await self.llm.acomplete(prompt)
        
        return StopEvent(result=str(response))