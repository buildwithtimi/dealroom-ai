# backend/app/engine/02_generation/workflow.py
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
import importlib

# 🟢 Import a customized event to bridge query rewriter to retriever safely
from llama_index.core.workflow import Event

class QueryRewriterEvent(Event):
    query: str
    original_query: str

from .events import RetrieveEvent

# Dynamically pull the evaluation module to keep things robust against numerical paths
eval_module = importlib.import_module("backend.app.engine.03_evaluation.evaluator")
DealRoomEvaluator = eval_module.DealRoomEvaluator

class DealRoomWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.index = index
        # Fixed the initialization argument mapping for consistency across files
        self.llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        Settings.llm = self.llm
        
        # Inject our automated evaluator gatekeeper
        self.evaluator = DealRoomEvaluator()

    @step
    async def rewrite_query(self, ev: StartEvent) -> QueryRewriterEvent:
        """
        🟢 Query Rewriting Step:
        Analyzes the user's input and reformulates it into a high-density 
        search variant for optimized vector similarity parsing.
        """
        raw_query = ev.get("query")
        if not raw_query:
            raise ValueError("Workflow started without a valid 'query'.")

        prompt = f"""
        You are an elite search optimization engine for a company due-diligence system.
        Your task is to rewrite the user's input question into a single, highly dense 
        search query optimized for vector-store retrieval. Focus on key financial terms, 
        metrics, or technical transformations mentioned.

        Do not answer the question. Only return the optimized search query string.

        Original User Input: {raw_query}
        Optimized Vector Search Query:
        """
        
        print(f"🔄 Optimizing raw prompt semantics: '{raw_query}'")
        rewritten_response = await self.llm.acomplete(prompt)
        optimized_query = str(rewritten_response).strip().strip('"').strip("'")
        print(f"🎯 Rewritten Search Target: '{optimized_query}'")
        
        return QueryRewriterEvent(query=optimized_query, original_query=raw_query)

    @step
    async def retrieve(self, ev: QueryRewriterEvent) -> RetrieveEvent:
        """
        Pipes the optimized query from the rewriter into the index retriever.
        """
        print(f"🔍 Searching Vector Store for: '{ev.query}'")
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query)
        # Pass both queries forward so synthesis knows what the user originally asked!
        return RetrieveEvent(query=ev.original_query, nodes=nodes)

    @step
    async def synthesize(self, ev: RetrieveEvent) -> StopEvent:
        context_str = "\n\n".join([node.node.get_content() for node in ev.nodes])
        
        prompt = f"""
        You are an advanced DealRoom AI due diligence assistant. 
        Analyze the provided context and answer the user query professionally.
        
        Context:
        {context_str}
        
        Query: {ev.query}
        Answer:
        """
        
        print("🧠 Synthesizing raw response via Gemini...")
        response = await self.llm.acomplete(prompt)
        response_str = str(response)
        
        # Route the response through our Evaluation Layer before passing back the output
        eval_report = await self.evaluator.evaluate_response(
            query=ev.query,
            response_str=response_str,
            contexts=ev.nodes
        )
        
        if not eval_report["success"]:
            print(f"⚠️ Guard Gate Triggered! Quality check failed. Feedback: {eval_report['feedback']}")
            final_output = f"[EVALUATION FAILED] \nReason: {eval_report['feedback']}\n\nRaw Answer:\n{response_str}"
        else:
            print("✅ Evaluation Passed! No hallucinations detected. Response is highly relevant.")
            final_output = response_str
        
        return StopEvent(result=final_output)