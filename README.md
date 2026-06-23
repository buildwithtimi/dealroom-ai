# DealRoom AI 🚀

**DealRoom AI** is a production-grade, full-stack Agentic Multi-Document RAG 
platform designed to automate venture capital and investment due diligence. 

By orchestrating event-driven **LlamaIndex Workflows**, the system ingests complex 
unstructured portfolios (such as pitch decks and legal agreements) alongside raw 
tabular data (like financial cap tables and burn-rate CSVs). 

Operating inside an isolated, containerized environment, the autonomous AI agent 
routes queries dynamically, self-corrects retrieval mismatches, and utilizes a 
secure Python code-execution sandbox to run real-time quantitative calculations—delivering 
fully grounded, hallucination-checked market insights.

### Key Architecture Pillars
* **AI Engine:** LlamaIndex Workflows, Hybrid Search (Vector + BM25), & Reranking.
* **Backend API:** Asynchronous FastAPI web server wrapped in Docker.
* **Frontend UI:** Streamlit interface tracking live agent execution states.
* **Observability:** Built-in production telemetry tracking via Langfuse.
*
