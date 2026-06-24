# backend/app/engine/02_generation/events.py
from typing import List
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class RetrieveEvent(Event):
    """Fired when documents have been successfully fetched from the vector store."""
    query: str
    nodes: List[NodeWithScore]

class EvaluationEvent(Event):
    """Fired after checking context quality. Signals if we need to rewrite or generate."""
    query: str
    nodes: List[NodeWithScore]
    is_sufficient: bool
    feedback: str = ""