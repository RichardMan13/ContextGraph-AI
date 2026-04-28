"""
src/chains/vector_chain.py
──────────────────────────
Stage 2: Takes the user query and candidate_ids from Stage 1,
then executes a semantic search on pgvector, keeping the top K results.
"""

import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableLambda
from src.tools.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _search_vector(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Receives 'query' and 'candidate_ids' from the previous chain state,
    runs semantic search, and stores the retrieved context into the state.
    """
    query = state.get("query", "")
    candidate_ids = state.get("candidate_ids", None)

    logger.info("Vector Stage: executing semantic search...")
    if candidate_ids is not None:
        logger.info(
            f"Vector Stage: Limiting to {len(candidate_ids)} candidates from Graph Stage."
        )

    retriever = VectorRetriever()

    # We retrieve the top 10 semantically matching plots within the graph intersection
    docs = retriever.search(query=query, candidate_ids=candidate_ids, top_k=10)

    # Format documents for the prompt
    vector_context = "\n\n".join(
        [
            f"- Movie: {d.metadata.get('title', 'Unknown')} | Semantic Relevance: {d.metadata.get('semantic_distance'):.3f}\n"
            f"  Plot: {d.page_content}"
            for d in docs
        ]
    )

    graph_context = state.get("graph_context", "")

    # Combine both structured graph data and semantic vector data
    full_context = ""
    if graph_context:
        full_context += (
            f"## STRUCTURED KNOWLEDGE FROM WATCHLIST (Graph):\n{graph_context}\n\n"
        )

    if vector_context:
        full_context += f"## SEMANTICALLY SIMILAR PLOTS (Vector):\n{vector_context}"

    if not full_context:
        full_context = "No movies or patterns found in your history matching these specific criteria."

    state["context"] = full_context
    state["docs"] = docs

    return state


def get_vector_chain():
    """
    Returns an LCEL Runnable that executes the vector retrieval.
    """
    return RunnableLambda(_search_vector)
