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

    # We retrieve the top 5 semantically matching plots within the graph intersection
    docs = retriever.search(query=query, candidate_ids=candidate_ids, top_k=5)

    # Format documents for the prompt
    context_str = "\n\n".join(
        [
            f"- Filme ID: {d.metadata.get('const')} | Relevância Semântica: {d.metadata.get('semantic_distance'):.3f}\n"
            f"  Sinopse: {d.page_content}"
            for d in docs
        ]
    )

    if not context_str:
        context_str = "Nenhum filme recuperado nos critérios."

    state["context"] = context_str
    state["docs"] = docs

    return state


def get_vector_chain():
    """
    Returns an LCEL Runnable that executes the vector retrieval.
    """
    return RunnableLambda(_search_vector)
