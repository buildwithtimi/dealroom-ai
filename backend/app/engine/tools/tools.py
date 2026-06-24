# backend/app/engine/tools/tools.py
import asyncio
import importlib
from llama_index.core.tools import FunctionTool
from llama_index.core.indices.vector_store import VectorStoreIndex

# Dynamically import the generation workflow to keep boundaries clean
gen_module = importlib.import_module("backend.app.engine.02_generation.workflow")
DealRoomWorkflow = gen_module.DealRoomWorkflow

def get_dealroom_workflow_tool(index: VectorStoreIndex) -> FunctionTool:
    """
    Wraps the evaluated DealRoom event-driven workflow into a structured FunctionTool 
    that an AI Agent can dynamically choose to invoke.
    """
    
    # Define the synchronous bridge execution for the agent framework
    def query_dealroom_vault(query: str) -> str:
        """
        Queries the DealRoom knowledge base to retrieve financial metrics, 
        burn rates, and pitch deck intelligence. Fully evaluated for accuracy.
        """
        workflow = DealRoomWorkflow(index=index)
        # Run the async state machine inside a clean synchronous loop wrapper
        return asyncio.run(workflow.run(query=query))

    # Return it wrapped as a LlamaIndex native tool container
    return FunctionTool.from_defaults(
        fn=query_dealroom_vault,
        name="query_dealroom_vault",
        description=(
            "Use this tool to search through company pitch documents, financial files, "
            "burn rates, and corporate transformations. Input should be a specific due-diligence question."
        )
    )