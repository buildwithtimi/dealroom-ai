# backend/app/engine/02_generation/workflow.py
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
import importlib

from .events import RetrieveEvent

# Dynamically pull the evaluation module to keep things robust against numerical paths
eval_module = importlib.import_module("backend.app.engine.03_evaluation.evaluator")
DealRoomEvaluator = eval_module.DealRoomEvaluator

class DealRoomWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.index = index
        self.llm = GoogleGenAI(
            model_name="models/gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        Settings.llm = self.llm
        
        # 🟢 Inject our new automated evaluator gatekeeper
        self.evaluator = DealRoomEvaluator()

    @step
    async def retrieve(self, ev: StartEvent) -> RetrieveEvent:
        query = ev.get("query")
        if not query:
            raise ValueError("Workflow started without a valid 'query'.")
            
        print(f"🔍 Searching Vector Store for: '{query}'")
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query)
        return RetrieveEvent(query=query, nodes=nodes)

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
        
        # 🟢 Route the response through our Evaluation Layer before passing back the output
        eval_report = await self.evaluator.evaluate_response(
            query=ev.query,
            response_str=response_str,
            contexts=ev.nodes
        )
        
        if not eval_report["success"]:
            print(f"⚠️ Guard Gate Triggered! Quality check failed. Feedback: {eval_report['feedback']}")
            # In a fully layered loop we could trigger query rewriting here, 
            # for now, we pass the warning out with the payload.
            final_output = f"[EVALUATION FAILED] \nReason: {eval_report['feedback']}\n\nRaw Answer:\n{response_str}"
        else:
            print("✅ Evaluation Passed! No hallucinations detected. Response is highly relevant.")
            final_output = response_str
        
        return StopEvent(result=final_output)