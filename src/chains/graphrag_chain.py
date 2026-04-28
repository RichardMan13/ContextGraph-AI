"""
src/chains/graphrag_chain.py
────────────────────────────
The orchestrator. Combines Graph (Stage 1), Vector (Stage 2), and Generation (Stage 3).
This is the only chain that the external API/UI interfaces with.
"""

from typing import Dict, Any

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.models.llm import get_llm
from src.chains.graph_chain import get_graph_chain
from src.chains.vector_chain import get_vector_chain
from src.prompts.answer_prompt import ANSWER_PROMPT


def get_graphrag_chain():
    """
    Builds the full GraphRAG LCEL chain:
        Dict{"query"}
          -> graph_chain (returns Dict{"query", "candidate_ids"})
          -> vector_chain (returns Dict{"query", "candidate_ids", "context", "docs"})
          -> ANSWER_PROMPT (formats string using "context" and "query")
          -> GPT-4o-mini
          -> StrOutputParser
    """

    # We pass the query through to the graph chain, but wait, the graph chain
    # currently expects {"query": ...} and returns {"cypher_query": ..., "candidate_ids": ...}.
    # We need to preserve the original query!

    def preserve_query(state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures the original 'query' key is kept in state after Stage 1."""
        query = state["query"]
        return {"query": query}

    # We use RunnablePassthrough.assign to keep 'query' while mapping new keys
    # from the graph_chain output.

    # Let's cleanly define the Stage 1 + Map
    stage_1 = RunnablePassthrough.assign(graph_data=get_graph_chain()) | (
        lambda x: {
            "query": x["query"],
            "cypher_query": x["graph_data"]["cypher_query"],
            "candidate_ids": x["graph_data"]["candidate_ids"],
            "graph_context": x["graph_data"]["graph_context"],
        }
    )

    # Stage 2 retrieves vector context
    stage_2 = get_vector_chain()

    # Stage 3 Answers
    stage_3 = ANSWER_PROMPT | get_llm(temperature=0.3) | StrOutputParser()

    # We run Stage 1, Stage 2, and then fork: Stage 3 gets the state to render answer.
    # The final output will be just the string, unless we want to return sources.
    # We will return a dict with both answer and sources.

    def assemble_final_output(state: Dict[str, Any]) -> Dict[str, Any]:
        answer = stage_3.invoke(state)
        return {
            "answer": answer,
            "cypher_query": state.get("cypher_query"),
            "candidate_ids": state.get("candidate_ids"),
            "graph_context": state.get("graph_context"),
            "source_documents": state.get("docs", []),
        }

    full_chain = stage_1 | stage_2 | assemble_final_output
    return full_chain


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    chain = get_graphrag_chain()

    print("----- GRAPH RAG TEST -----")
    res = chain.invoke(
        {"query": "Recommend a Science Fiction movie by Christopher Nolan"}
    )
    print("\n[CYPHER]")
    print(res["cypher_query"])
    print("\n[ANSWER]")
    print(res["answer"])
