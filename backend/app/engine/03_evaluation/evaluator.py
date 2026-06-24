# backend/app/engine/03_evaluation/evaluator.py
import asyncio
import os
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
# 🟢 Switched to the modern non-deprecated LLM module
from llama_index.llms.google_genai import GoogleGenAI
# 🟢 Added ServiceUnavailable to handle 503 demand spikes gracefully
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

class DealRoomEvaluator:
    def __init__(self):
        # 🟢 Initialized using the updated SDK pattern
        self.judge_llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        self.faithfulness = FaithfulnessEvaluator(llm=self.judge_llm)
        self.relevancy = RelevancyEvaluator(llm=self.judge_llm)

    async def _execute_with_retry(self, evaluator, **kwargs):
        """Helper to execute an evaluation metric with backoff logic for 429s and 503s."""
        max_retries = 3
        delay = 10  # Initial wait window in seconds if throttled
        
        for attempt in range(max_retries):
            try:
                return await evaluator.aevaluate(**kwargs)
            # 🟢 Intercepting both quota limits (429) and high-demand server hiccups (503)
            except (ResourceExhausted, ServiceUnavailable) as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ {type(e).__name__} hit during evaluation loop. Backing off for {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponentially scale the wait window

    async def evaluate_response(self, query: str, response_str: str, contexts: list) -> dict:
        print("⚖️ Validator Guard active: Auditing response semantics...")
        raw_contexts = [node.node.get_content() for node in contexts]
        
        # 1. Run Faithfulness Check
        faith_result = await self._execute_with_retry(
            self.faithfulness,
            query=query,
            response=response_str,
            contexts=raw_contexts
        )
        
        print("⏳ Cool-down: Pacing between metric evaluations...")
        await asyncio.sleep(5.0)
        
        # 2. Run Relevancy Check
        rel_result = await self._execute_with_retry(
            self.relevancy,
            query=query,
            response=response_str,
            contexts=raw_contexts
        )
        
        # Extract values safely with fallbacks to satisfy Pylance
        faith_passed = getattr(faith_result, "passing", False)
        faith_feedback = getattr(faith_result, "feedback", "No feedback available")
        
        rel_passed = getattr(rel_result, "passing", False)
        rel_feedback = getattr(rel_result, "feedback", "No feedback available")
        
        is_passing = bool(faith_passed and rel_passed)
        
        return {
            "success": is_passing,
            "faithfulness_passed": faith_passed,
            "relevancy_passed": rel_passed,
            "feedback": f"Faithfulness: {faith_feedback} | Relevancy: {rel_feedback}"
        }
